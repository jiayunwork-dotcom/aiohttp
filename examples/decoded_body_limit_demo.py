#!/usr/bin/env python3
"""Demonstrate the optional decoded-body limit.

Run from the repository root with the project dependencies available:

    PYTHONPATH=. python3 examples/decoded_body_limit_demo.py
"""

from __future__ import annotations

import asyncio
import gzip
import time
import tracemalloc
from unittest import mock

from aiohttp import web
from aiohttp.http_parser import DeflateBuffer, HttpRequestParser, HttpPayloadParser
from aiohttp.streams import PayloadSizeError, StreamReader
from aiohttp.web_protocol import RequestPayloadError

LIMIT = 64 * 1024
DEFLATE_FIXTURE = b"x\x9cKI,I\x04\x00\x04\x00\x01\x9b"
PLAIN_FIXTURE = b"data"


def raw_message(
    body: bytes,
    encoding: str | None = None,
    chunked: bool = False,
    path: str = "/",
) -> bytes:
    headers = [f"POST {path} HTTP/1.1\r\nHost: demo\r\n".encode()]
    if encoding:
        headers.append(f"Content-Encoding: {encoding}\r\n".encode())
    if chunked:
        headers.append(b"Transfer-Encoding: chunked\r\n\r\n")
        frame = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
    else:
        headers.append(f"Content-Length: {len(body)}\r\n\r\n".encode())
        frame = body
    return b"".join(headers) + frame


def feed_parser(raw: bytes, shard_size: int, limit: int | None) -> StreamReader:
    loop = asyncio.get_running_loop()
    parser = HttpRequestParser(
        mock.Mock(),
        loop,
        2**16,
        payload_exception=RequestPayloadError,
        decoded_body_limit=limit,
    )
    payload: StreamReader | None = None
    for pos in range(0, len(raw), shard_size):
        messages, _, _ = parser.feed_data(raw[pos : pos + shard_size])
        for _, candidate in messages:
            payload = candidate
        if payload and isinstance(payload.exception(), PayloadSizeError):
            break
    assert payload is not None
    return payload


async def wait_until(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("timed out waiting for condition")


def mock_transport() -> mock.Mock:
    transport = mock.Mock(spec=asyncio.Transport)
    transport.is_closing.return_value = False
    transport.get_write_buffer_size.return_value = 0
    return transport


def output_bytes(transport: mock.Mock) -> bytes:
    return b"".join(
        call.args[0]
        for call in transport.write.call_args_list
        if call.args and isinstance(call.args[0], (bytes, bytearray))
    )


async def protocol_for(handler, limit: int = 4):
    server = web.Server(handler, decoded_body_limit=limit, lingering_time=0)
    protocol = server()
    transport = mock_transport()
    protocol.connection_made(transport)
    return protocol, transport


def demo_configuration() -> None:
    app = web.Application(handler_args={"decoded_body_limit": LIMIT})
    handler = app._make_handler()
    assert handler._kwargs["decoded_body_limit"] == LIMIT
    print(f"[1] Application handler_args -> RequestHandler: {LIMIT} bytes")


async def demo_splits() -> None:
    cases = [
        ("deflate", DEFLATE_FIXTURE),
        ("gzip", gzip.compress(PLAIN_FIXTURE)),
    ]
    checked = 0
    for encoding, compressed in cases:
        for chunked in (False, True):
            raw = raw_message(compressed, encoding, chunked)
            for shard_size in (1, 2, 3, 7, len(raw)):
                allowed = feed_parser(raw, shard_size, 4)
                assert await allowed.read() == PLAIN_FIXTURE
                rejected = feed_parser(raw, shard_size, 3)
                assert isinstance(rejected.exception(), PayloadSizeError)
                checked += 1
    print(f"[2] Content-Length/chunked split checks passed: {checked} cases")


async def demo_memory_bomb() -> None:
    zeros = b"\0" * (64 * 1024 * 1024)
    compressed = gzip.compress(zeros, compresslevel=9)
    del zeros

    loop = asyncio.get_running_loop()
    reader = StreamReader(
        mock.Mock(), 2**16, loop=loop, decoded_body_limit=LIMIT
    )
    decoder = DeflateBuffer(reader, "gzip")

    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        decoder.feed_data(compressed, len(compressed))
    except PayloadSizeError as exc:
        assert exc.limit == LIMIT
    else:
        raise AssertionError("bomb was not rejected")
    _, peak = tracemalloc.get_traced_memory()
    extra_peak = peak - baseline
    tracemalloc.stop()

    assert extra_peak <= 2 * 1024 * 1024
    print(
        "[3] 64 MiB zero bomb: "
        f"compressed={len(compressed)} B, buffered={reader.total_bytes} B, "
        f"tracemalloc peak={extra_peak / 1024:.1f} KiB"
    )


async def demo_413_and_pipeline() -> None:
    woke: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()

    async def reading_handler(request: web.Request) -> web.Response:
        try:
            await request.content.read()
        except BaseException as exc:
            woke.set_result(exc)
            raise
        return web.Response()

    protocol, transport = await protocol_for(reading_handler, limit=4)
    task = protocol._task_handler
    assert task is not None
    protocol.data_received(raw_message(b"abcde"))
    exc = await asyncio.wait_for(woke, 2)
    await asyncio.wait_for(task, 2)
    output = output_bytes(transport)
    assert isinstance(exc, PayloadSizeError)
    assert b"HTTP/1.1 413" in output and b"Connection: close" in output
    print(
        "[4] waiting reader got",
        type(exc).__name__,
        "and mock transport wrote 413 + close",
    )

    seen: list[bytes] = []

    async def pipeline_handler(request: web.Request) -> web.Response:
        seen.append(await request.read())
        return web.Response()

    protocol, transport = await protocol_for(pipeline_handler, limit=4)
    first = raw_message(b"abcd", path="/one")
    second = raw_message(b"efgh", path="/two")
    protocol.data_received(first + second)
    await wait_until(lambda: len(seen) == 2)
    assert seen == [b"abcd", b"efgh"]
    assert output_bytes(transport).count(b"HTTP/1.1 200") == 2
    print("[6a] two pipelined requests each received their own 4-byte quota")

    calls: list[str] = []

    async def oversize_handler(request: web.Request) -> web.Response:
        calls.append(request.path)
        await request.read()
        return web.Response()

    protocol, transport = await protocol_for(oversize_handler, limit=4)
    task = protocol._task_handler
    assert task is not None
    bad = raw_message(b"abcde", path="/bad")
    trailing = raw_message(b"efgh", path="/trailing")
    protocol.data_received(bad + trailing)
    await asyncio.wait_for(task, 2)
    protocol.data_received(raw_message(b"ijkl", path="/later"))
    await asyncio.sleep(0)
    assert calls == ["/bad"]
    output = output_bytes(transport)
    assert output.count(b"HTTP/1.1 413") == 1
    print(
        "[6b] over-limit request served 413; "
        "spliced/later requests dispatch count = 0"
    )


async def run_benchmark(compressed: bytes, shard_size: int, expected: int):
    loop = asyncio.get_running_loop()
    reader = StreamReader(
        mock.Mock(), 2**16, loop=loop, decoded_body_limit=expected * 2
    )
    parser = HttpPayloadParser(
        reader,
        length=len(compressed),
        compression="gzip",
        decoded_body_limit=expected * 2,
    )
    started = time.perf_counter()
    for pos in range(0, len(compressed), shard_size):
        parser.feed_data(compressed[pos : pos + shard_size])
    elapsed = time.perf_counter() - started
    chunk_count = len(reader._buffer)
    data = await reader.read()
    assert len(data) == expected
    return elapsed, chunk_count


async def demo_shard_cost() -> None:
    plain = b"".join(
        f"line {i:06d} quick brown fox {i % 11}\n".encode() for i in range(25_000)
    )
    compressed = gzip.compress(plain, compresslevel=6)
    print("[8] same gzip body, smaller network shards:")
    print(f"    plain={len(plain)} B compressed={len(compressed)} B")
    print("    shard_B  shards   best_ms  output_chunks")
    for shard_size in (len(compressed), 4096, 1024, 256, 64, 16, 1):
        await run_benchmark(compressed, shard_size, len(plain))
        times = []
        for _ in range(3):
            elapsed, chunks = await run_benchmark(
                compressed, shard_size, len(plain)
            )
            times.append(elapsed)
        shard_count = (len(compressed) + shard_size - 1) // shard_size
        print(
            f"    {shard_size:7d} {shard_count:7d} "
            f"{min(times) * 1000:8.2f} {chunks:13d}"
        )


async def main() -> None:
    demo_configuration()
    await demo_splits()
    await demo_memory_bomb()
    await demo_413_and_pipeline()
    await demo_shard_cost()
    print("demo completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
