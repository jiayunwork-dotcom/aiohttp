"""Terminal demonstration of request-scoped allow_networks.

Run from a checkout with aiohttp's dependencies available, for example::

    PYTHONPATH=. python examples/client_allow_networks.py
"""

from __future__ import annotations

import asyncio
import socket
import ssl
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

from yarl import URL

from aiohttp import ClientSession, TCPConnector
from aiohttp.client_exceptions import ClientAddressNotAllowedError
from aiohttp.client_reqrep import ClientRequest
from aiohttp.connector import Connection


def resolve_result(ip: str, port: int, host: str) -> Dict[str, Any]:
    return {
        "hostname": host,
        "host": ip,
        "port": port,
        "family": socket.AF_INET,
        "proto": 0,
        "flags": 0,
    }


class FakeResolver:
    def __init__(self, addresses: List[str], gate: Optional[asyncio.Event] = None):
        self.addresses = addresses
        self.gate = gate
        self.calls = 0

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_UNSPEC):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        return [resolve_result(ip, port, host) for ip in self.addresses]


class RawHTTP:
    def __init__(self, *, https: bool = False) -> None:
        self.https = https
        self.server: Optional[asyncio.AbstractServer] = None
        self.port = 0
        self.accepted = 0
        self.paths: List[str] = []

    async def start(self) -> "RawHTTP":
        ssl_context: Optional[ssl.SSLContext] = None
        if self.https:
            import trustme

            ca = trustme.CA()
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ca.issue_cert("127.0.0.1", "localhost").configure_cert(ssl_context)

        self.server = await asyncio.start_server(
            self.handle, "127.0.0.1", 0, ssl=ssl_context
        )
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def close(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    @property
    def url(self) -> str:
        scheme = "https" if self.https else "http"
        return f"{scheme}://127.0.0.1:{self.port}"

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.accepted += 1
        try:
            while True:
                headers = await reader.readuntil(b"\r\n\r\n")
                if not headers:
                    break
                first = headers.split(b"\r\n", 1)[0].decode()
                path = first.split(" ")[1]
                self.paths.append(path)

                if path == "/redirect-local":
                    payload = (
                        b"HTTP/1.1 302 Found\r\nLocation: /final\r\n"
                        b"Content-Length: 0\r\n\r\n"
                    )
                elif path == "/redirect-denied":
                    payload = (
                        b"HTTP/1.1 302 Found\r\n"
                        b"Location: http://127.0.0.2:9/forbidden\r\n"
                        b"Content-Length: 0\r\n\r\n"
                    )
                else:
                    payload = (
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                        b"Connection: keep-alive\r\n\r\nok"
                    )
                writer.write(payload)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


def line(title: str) -> None:
    print(f"\n=== {title} ===")


async def demo_literal_and_empty(server: RawHTTP) -> None:
    line("1. Literal IP, CIDR deny, and explicit empty deny-all")
    async with ClientSession() as session:
        async with session.get(
            f"{server.url}/literal-ok", allow_networks=["127.0.0.1/32"]
        ) as resp:
            print(f"127.0.0.1/32  -> HTTP {resp.status}, body={await resp.text()!r}")

        for networks in (["10.0.0.0/8"], []):
            try:
                async with session.get(
                    f"{server.url}/literal-no", allow_networks=networks
                ):
                    raise RuntimeError("request unexpectedly succeeded")
            except ClientAddressNotAllowedError as exc:
                print(f"{str(networks):18}-> {type(exc).__name__}: {exc}")


async def demo_dns(server: RawHTTP) -> None:
    line("2/5. Shared/coalesced DNS cache filters per request")
    gate = asyncio.Event()
    resolver = FakeResolver(["127.0.0.1", "127.0.0.2"], gate)
    connector = TCPConnector(resolver=resolver)
    try:
        task_a = asyncio.create_task(
            connector._resolve_host(
                "mixed.example", 80, allow_networks=["127.0.0.1/32"]
            )
        )
        task_b = asyncio.create_task(
            connector._resolve_host(
                "mixed.example", 80, allow_networks=["127.0.0.2/32"]
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        print(f"concurrent resolver calls before release: {resolver.calls}")
        gate.set()
        a, b = await asyncio.gather(task_a, task_b)
        print("policy .1/32 ->", [x["host"] for x in a])
        print("policy .2/32 ->", [x["host"] for x in b])

        attempted: List[str] = []
        original = connector._wrap_create_connection

        async def spy(factory: Any, **kwargs: Any) -> Any:
            attempted.extend(info[4][0] for info in kwargs["addr_infos"])
            return await original(factory, **kwargs)

        connector._wrap_create_connection = spy  # type: ignore[method-assign]
        async with ClientSession(connector=connector) as session:
            async with session.get(
                f"http://mixed.example:{server.port}/dns-only-allowed",
                allow_networks=["127.0.0.1/32"],
            ) as resp:
                print(f"mixed DNS HTTP -> {resp.status}, body={await resp.text()!r}")
        print("dialled candidates:", attempted)
    finally:
        await connector.close()


async def demo_pool(server: RawHTTP) -> None:
    line("3/7. Pool peer check and same-policy reuse")
    connector = TCPConnector(keepalive_timeout=30)
    try:
        async with ClientSession(connector=connector) as session:
            async with session.get(
                f"{server.url}/pool-first", allow_networks=["127.0.0.0/8"]
            ) as resp:
                await resp.read()
            try:
                async with session.get(
                    f"{server.url}/pool-second-denied",
                    allow_networks=["10.0.0.0/8"],
                ):
                    raise RuntimeError("pooled peer unexpectedly accepted")
            except ClientAddressNotAllowedError as exc:
                print(f"narrow policy -> {type(exc).__name__}")
        print(
            "denied pooled request path reached server: "
            f"{'/pool-second-denied' in server.paths}"
        )
        print(f"all server paths observed by demo: {server.paths}")
    finally:
        await connector.close()

    reuse_server = await RawHTTP().start()
    try:
        async with ClientSession() as session:
            for path in ("/reuse-1", "/reuse-2"):
                async with session.get(
                    f"{reuse_server.url}{path}",
                    allow_networks=["127.0.0.1/32"],
                ) as resp:
                    await resp.read()
        print(
            f"same policy requests={reuse_server.paths}, "
            f"accepted TCP connections={reuse_server.accepted}"
        )
    finally:
        await reuse_server.close()


async def demo_redirects(server: RawHTTP) -> None:
    line("4. Redirects retain the initial allow_networks")
    async with ClientSession() as session:
        async with session.get(
            f"{server.url}/redirect-local",
            allow_networks=["127.0.0.1/32"],
        ) as resp:
            print(f"same-host redirect -> {resp.status} {resp.url.path}")

        try:
            async with session.get(
                f"{server.url}/redirect-denied",
                allow_networks=["127.0.0.1/32"],
            ):
                raise RuntimeError("redirect to forbidden IP unexpectedly succeeded")
        except ClientAddressNotAllowedError as exc:
            print(f"cross-IP redirect -> {type(exc).__name__}: host={exc.host}")


async def demo_https() -> None:
    line("1b. HTTPS uses the same transport peername check")
    try:
        import trustme  # noqa: F401
    except ImportError:
        print("skipped: install 'trustme' to generate a local certificate")
        return

    server = await RawHTTP(https=True).start()
    connector = TCPConnector()
    try:
        async with ClientSession(connector=connector) as session:
            async with session.get(
                f"{server.url}/https-ok",
                ssl=False,
                allow_networks=["127.0.0.1/32"],
            ) as resp:
                print(f"HTTPS allowed -> {resp.status}, body={await resp.text()!r}")
            try:
                async with session.get(
                    f"{server.url}/https-denied",
                    ssl=False,
                    allow_networks=[],
                ):
                    raise RuntimeError("empty policy unexpectedly allowed HTTPS")
            except ClientAddressNotAllowedError as exc:
                print(f"HTTPS empty list -> {type(exc).__name__}")
    finally:
        await connector.close()
        await server.close()


async def demo_send_backstop() -> None:
    line("6. ClientRequest.send rechecks the final peer before headers")
    loop = asyncio.get_running_loop()
    connector = TCPConnector()
    req = ClientRequest(
        "GET",
        URL("http://example.test/"),
        loop=loop,
        allow_networks=["10.0.0.0/8"],
    )
    protocol = Mock()
    protocol.transport.get_extra_info.side_effect = lambda name: (
        ("127.0.0.1", 4433) if name == "peername" else None
    )
    conn = Connection(connector, req.connection_key, protocol, loop)  # type: ignore[arg-type]
    try:
        await req.send(conn)
    except ClientAddressNotAllowedError as exc:
        print(f"send() -> {type(exc).__name__}: {exc}")
    finally:
        conn.close()
        await connector.close()


async def main() -> None:
    server = await RawHTTP().start()
    try:
        await demo_literal_and_empty(server)
        await demo_https()
        await demo_dns(server)
        await demo_pool(server)
        await demo_redirects(server)
        await demo_send_backstop()
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())
