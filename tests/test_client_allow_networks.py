from __future__ import annotations

import asyncio
import contextlib
import socket
import ssl
from ipaddress import IPv4Network
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, Mock

import pytest
from yarl import URL

import aiohttp
from aiohttp import ClientRequest, ClientSession, TCPConnector
from aiohttp.client_exceptions import ClientAddressNotAllowedError
from aiohttp.client_reqrep import StreamWriter as RequestStreamWriter
from aiohttp.networks import NetworkPolicy


def result(ip: str, port: int, host: str = "mixed.example") -> Dict[str, Any]:
    return {
        "hostname": host,
        "host": ip,
        "port": port,
        "family": socket.AF_INET,
        "proto": 0,
        "flags": 0,
    }


class FakeResolver:
    def __init__(self, addresses: List[str], *, gate: Optional[asyncio.Event] = None):
        self.addresses = addresses
        self.gate = gate
        self.calls = 0

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_UNSPEC):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        return [result(ip, port, host) for ip in self.addresses]


class RawHTTPServer:
    """Minimal keep-alive HTTP/1.1 server that records raw chunks."""

    def __init__(
        self,
        *,
        https: bool = False,
        host: str = "127.0.0.1",
        redirect_location: Optional[str] = None,
    ):
        self.https = https
        self.host = host
        self.redirect_location = redirect_location
        self.server: Optional[asyncio.AbstractServer] = None
        self.port = 0
        self.accepted = 0
        self.chunks: List[bytes] = []
        self.request_paths: List[str] = []
        self.writers: List[asyncio.StreamWriter] = []
        self.tasks: Set[asyncio.Task[None]] = set()
        self._ssl_context: Optional[ssl.SSLContext] = None

    async def start(self) -> "RawHTTPServer":
        if self.https:
            import trustme

            ca = trustme.CA()
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ca.issue_cert("127.0.0.1", "localhost").configure_cert(ctx)
            self._ssl_context = ctx

        self.server = await asyncio.start_server(
            self._handle,
            self.host,
            0,
            ssl=self._ssl_context,
        )
        sockets = self.server.sockets
        assert sockets is not None
        self.port = sockets[0].getsockname()[1]
        return self

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

        for writer in list(self.writers):
            writer.close()

        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)

        for writer in list(self.writers):
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    @property
    def url(self) -> str:
        scheme = "https" if self.https else "http"
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{scheme}://{host}:{self.port}"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.accepted += 1
        task = asyncio.current_task()
        if task is not None:
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        self.writers.append(writer)
        buffer = b""
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                self.chunks.append(chunk)
                buffer += chunk

                while b"\r\n\r\n" in buffer:
                    headers, buffer = buffer.split(b"\r\n\r\n", 1)
                    request_line = headers.split(b"\r\n", 1)[0].decode("latin1")
                    parts = request_line.split(" ")
                    path = parts[1] if len(parts) >= 2 else "/"
                    self.request_paths.append(path)

                    if path.startswith("/redirect"):
                        location = self.redirect_location or "/final"
                        body = b""
                        head = (
                            "HTTP/1.1 302 Found\r\n"
                            f"Location: {location}\r\n"
                            f"Content-Length: {len(body)}\r\n"
                            "Connection: keep-alive\r\n\r\n"
                        ).encode("latin1")
                        writer.write(head + body)
                    else:
                        body = b"ok"
                        writer.write(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Length: 2\r\n"
                            b"Connection: keep-alive\r\n\r\n" + body
                        )
                    await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


async def test_none_and_empty_list_have_different_meanings() -> None:
    unrestricted = NetworkPolicy.parse(None)
    deny_all = NetworkPolicy.parse([])

    assert unrestricted.enabled is False
    assert unrestricted.allows_address("127.0.0.1") is True
    assert deny_all.enabled is True
    assert deny_all.networks == ()
    assert deny_all.allows_address("127.0.0.1") is False

    with pytest.raises(ValueError):
        NetworkPolicy.parse(["not-a-cidr"])
    with pytest.raises(TypeError):
        NetworkPolicy.parse("127.0.0.0/8")

    ipv4 = NetworkPolicy.parse(["127.0.0.0/8", IPv4Network("10.0.0.0/8")])
    assert ipv4.allows_address("127.1.2.3")
    assert ipv4.allows_address("127.0.0.1.")
    assert ipv4.allows_address("10.1.2.3")
    assert not ipv4.allows_address("192.168.1.1")
    # IPv4-mapped IPv6 peer names are normalized before matching.
    assert ipv4.allows_address("::ffff:127.0.0.1")

    ipv6 = NetworkPolicy.parse(["::1/128", IPv4Network("127.0.0.0/8")])
    assert ipv6.allows_address("::1")
    assert not ipv6.allows_address("2001:db8::1")
    assert ipv6.allows_address("127.0.0.1")


async def test_literal_ipv4_request_is_allowed_and_denied() -> None:
    server = await RawHTTPServer().start()
    connector = TCPConnector()
    try:
        async with ClientSession(connector=connector) as session:
            async with session.get(
                f"{server.url}/allowed",
                allow_networks=["127.0.0.1/32"],
            ) as resp:
                assert resp.status == 200
                assert await resp.text() == "ok"

            with pytest.raises(ClientAddressNotAllowedError):
                async with session.get(
                    f"{server.url}/denied",
                    allow_networks=["10.0.0.0/8"],
                ):
                    pass

            with pytest.raises(ClientAddressNotAllowedError):
                async with session.get(f"{server.url}/deny-all", allow_networks=[]):
                    pass

        assert server.request_paths == ["/allowed"]
    finally:
        await connector.close()
        await server.close()


async def test_literal_ipv6_request_is_allowed_and_denied() -> None:
    try:
        server = await RawHTTPServer(host="::1").start()
    except OSError:
        pytest.skip("IPv6 loopback is not available")

    connector = TCPConnector()
    try:
        async with ClientSession(connector=connector) as session:
            async with session.get(
                f"{server.url}/ipv6-allowed",
                allow_networks=["::1/128"],
            ) as resp:
                assert resp.status == 200
                assert await resp.text() == "ok"

            with pytest.raises(ClientAddressNotAllowedError):
                async with session.get(
                    f"{server.url}/ipv6-denied",
                    allow_networks=["2001:db8::/32"],
                ):
                    pass

        assert server.request_paths == ["/ipv6-allowed"]
    finally:
        await connector.close()
        await server.close()


async def test_https_uses_same_peer_policy() -> None:
    pytest.importorskip("trustme")
    server = await RawHTTPServer(https=True).start()
    connector = TCPConnector()
    try:
        async with ClientSession(connector=connector) as session:
            async with session.get(
                f"{server.url}/https-ok",
                ssl=False,
                allow_networks=[IPv4Network("127.0.0.1/32")],
            ) as resp:
                assert resp.status == 200
                assert await resp.text() == "ok"

            with pytest.raises(ClientAddressNotAllowedError):
                async with session.get(
                    f"{server.url}/https-denied",
                    ssl=False,
                    allow_networks=[],
                ):
                    pass

        assert server.request_paths == ["/https-ok"]
    finally:
        await connector.close()
        await server.close()


async def test_default_and_same_policy_connections_are_reused() -> None:
    server = await RawHTTPServer().start()
    try:
        async with ClientSession() as session:
            for path in ("/one", "/two"):
                async with session.get(f"{server.url}{path}") as resp:
                    assert await resp.text() == "ok"

        assert server.accepted == 1
        assert server.request_paths == ["/one", "/two"]

        server.accepted = 0
        server.request_paths.clear()
        async with ClientSession() as session:
            networks = ["127.0.0.1/32"]
            for path in ("/three", "/four"):
                async with session.get(
                    f"{server.url}{path}", allow_networks=networks
                ) as resp:
                    assert await resp.text() == "ok"

        assert server.accepted == 1
        assert server.request_paths == ["/three", "/four"]
    finally:
        await server.close()


async def test_pooled_peer_is_checked_before_more_request_bytes() -> None:
    server = await RawHTTPServer().start()
    connector = TCPConnector(keepalive_timeout=30)
    try:
        async with ClientSession(connector=connector) as session:
            async with session.get(
                f"{server.url}/first",
                allow_networks=["127.0.0.0/8"],
            ) as resp:
                await resp.read()

            # The response release returns the live connection to the pool.
            key = next(iter(connector._conns))
            assert connector._conns[key]

            with pytest.raises(ClientAddressNotAllowedError):
                async with session.get(
                    f"{server.url}/second",
                    allow_networks=["10.0.0.0/8"],
                ):
                    pass

            # Give an erroneous write, if any, a chance to arrive.
            await asyncio.sleep(0.1)

            # The fresh mismatched pooled connection is preserved for a
            # broader policy rather than poisoned or permanently removed.
            assert key in connector._conns
            assert len(connector._conns[key]) == 1

        assert server.request_paths == ["/first"]
        assert len(server.chunks) == 1
        assert server.chunks[0].startswith(b"GET /first")
    finally:
        await connector.close()
        await server.close()


async def test_request_send_rechecks_actual_peer_before_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    req = ClientRequest(
        "GET",
        URL("http://example.test/"),
        loop=loop,
        allow_networks=["10.0.0.0/8"],
    )

    connector = TCPConnector()

    class Transport:
        def get_extra_info(self, name: str) -> Any:
            if name == "peername":
                return ("127.0.0.1", 12345)
            return None

    protocol = type(
        "Protocol",
        (),
        {"transport": Transport(), "close": lambda self: None},
    )()
    conn = aiohttp.connector.Connection(connector, req.connection_key, protocol, loop)

    def forbidden_writer(*args: Any, **kwargs: Any) -> RequestStreamWriter:
        raise AssertionError("request writer must not be created after peer reject")

    monkeypatch.setattr("aiohttp.client_reqrep.StreamWriter", forbidden_writer)

    with pytest.raises(ClientAddressNotAllowedError) as exc_info:
        await req.send(conn)

    assert exc_info.value.address is not None
    assert "allow_networks" in str(exc_info.value)
    conn.close()
    await connector.close()


async def test_session_rechecks_connection_returned_by_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = TCPConnector()
    conn = Mock()
    conn.transport.get_extra_info.side_effect = lambda name: (
        ("127.0.0.1", 12345) if name == "peername" else None
    )
    connect = AsyncMock(return_value=conn)
    monkeypatch.setattr(connector, "connect", connect)

    try:
        async with ClientSession(connector=connector) as session:
            with pytest.raises(ClientAddressNotAllowedError):
                async with session.get(
                    "http://example.test/",
                    allow_networks=["10.0.0.0/8"],
                ):
                    pass

        conn.close.assert_called_once_with()
    finally:
        await connector.close()


async def test_dns_results_are_filtered_for_cache_hit_and_shared_cache() -> None:
    resolver = FakeResolver(["127.0.0.1", "127.0.0.2"])
    connector = TCPConnector(resolver=resolver)
    try:
        first = await connector._resolve_host(
            "mixed.example",
            80,
            allow_networks=["127.0.0.1/32"],
        )
        second = await connector._resolve_host(
            "mixed.example",
            80,
            allow_networks=["127.0.0.2/32"],
        )

        assert [item["host"] for item in first] == ["127.0.0.1"]
        assert [item["host"] for item in second] == ["127.0.0.2"]
        assert resolver.calls == 1

        # The shared cache itself still contains the complete result.
        cached = connector._cached_hosts.next_addrs(("mixed.example", 80))
        assert {item["host"] for item in cached} == {"127.0.0.1", "127.0.0.2"}
    finally:
        await connector.close()


async def test_concurrent_dns_coalescing_does_not_share_allow_decision() -> None:
    gate = asyncio.Event()
    resolver = FakeResolver(["127.0.0.1", "127.0.0.2"], gate=gate)
    connector = TCPConnector(resolver=resolver)
    try:
        task_one = asyncio.create_task(
            connector._resolve_host(
                "coalesced.example",
                80,
                allow_networks=["127.0.0.1/32"],
            )
        )
        task_two = asyncio.create_task(
            connector._resolve_host(
                "coalesced.example",
                80,
                allow_networks=["127.0.0.2/32"],
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert resolver.calls == 1

        gate.set()
        one, two = await asyncio.gather(task_one, task_two)
        assert [item["host"] for item in one] == ["127.0.0.1"]
        assert [item["host"] for item in two] == ["127.0.0.2"]
    finally:
        await connector.close()


async def test_mixed_dns_only_attempts_allowed_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = await RawHTTPServer().start()
    resolver = FakeResolver(["127.0.0.2", "127.0.0.1"])
    connector = TCPConnector(resolver=resolver)

    attempted: List[str] = []
    original = connector._wrap_create_connection

    async def wrap(factory: Any, **kwargs: Any) -> Any:
        for addr_info in kwargs["addr_infos"]:
            attempted.append(addr_info[4][0])
        assert all(ip == "127.0.0.1" for ip in attempted)
        return await original(factory, **kwargs)

    monkeypatch.setattr(connector, "_wrap_create_connection", wrap)

    try:
        async with ClientSession(connector=connector) as session:
            async with session.get(
                f"http://mixed.example:{server.port}/dns",
                allow_networks=["127.0.0.1/32"],
            ) as resp:
                assert resp.status == 200
                assert await resp.text() == "ok"

        assert attempted == ["127.0.0.1"]
        assert server.request_paths == ["/dns"]
    finally:
        await connector.close()
        await server.close()


async def test_dns_result_without_allowed_address_is_rejected() -> None:
    resolver = FakeResolver(["127.0.0.2"])
    connector = TCPConnector(resolver=resolver)
    try:
        async with ClientSession(connector=connector) as session:
            with pytest.raises(ClientAddressNotAllowedError) as exc_info:
                async with session.get(
                    "http://blocked.example/",
                    allow_networks=["127.0.0.1/32"],
                ):
                    pass
        assert exc_info.value.host == "blocked.example"
    finally:
        await connector.close()


class RecordingRequest(ClientRequest):
    seen_policies: List[NetworkPolicy] = []

    async def send(self, conn: Any) -> Any:
        self.seen_policies.append(self.network_policy)
        return await super().send(conn)


async def test_redirects_keep_initial_policy_object() -> None:
    server = await RawHTTPServer(redirect_location="/final").start()
    RecordingRequest.seen_policies = []
    try:
        async with ClientSession(request_class=RecordingRequest) as session:
            async with session.get(
                f"{server.url}/redirect",
                allow_redirects=True,
                allow_networks=["127.0.0.1/32"],
            ) as resp:
                assert resp.status == 200
                assert resp.url.path == "/final"
                assert len(resp.history) == 1

        assert server.request_paths == ["/redirect", "/final"]
        assert len(RecordingRequest.seen_policies) == 2
        first, second = RecordingRequest.seen_policies
        assert first is second
        assert first.allows_address("127.0.0.1")
        assert not first.allows_address("127.0.0.2")
    finally:
        await server.close()


async def test_redirect_cannot_drop_or_widen_policy() -> None:
    server = await RawHTTPServer(
        redirect_location="http://127.0.0.2:9/forbidden"
    ).start()
    try:
        async with ClientSession() as session:
            with pytest.raises(ClientAddressNotAllowedError) as exc_info:
                async with session.get(
                    f"{server.url}/redirect",
                    allow_redirects=True,
                    allow_networks=["127.0.0.1/32"],
                ):
                    pass

        assert exc_info.value.host == "127.0.0.2"
        assert server.request_paths == ["/redirect"]
    finally:
        await server.close()
