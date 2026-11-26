from __future__ import annotations

import asyncio
import gzip
import tracemalloc
import zlib
from typing import Awaitable, Callable, Optional
from unittest import mock

import pytest

from aiohttp import web
from aiohttp.http_parser import HttpRequestParser
from aiohttp.streams import PayloadSizeError, StreamReader
from aiohttp.web_protocol import RequestPayloadError

LOWMEM_LIMIT = 64 * 1024

# Repository fixture from tests/test_http_parser.py:
# c=compressobj(wbits=15); b''.join([c.compress(b'data'), c.flush()])
DEFLATE_DATA = b"x\x9cKI,I\x04\x00\x04\x00\x01\x9b"
PLAIN_DATA = b"data"


def frame_chunked(body: bytes, chunk_size: int = 3) -> bytes:
    parts = []
    for i in range(0, len(body), chunk_size):
        chunk = body[i : i + chunk_size]
        parts.append(f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n")
    parts.append(b"0\r\n\r\n")
    return b"".join(parts)


def raw_message(
    body: bytes,
    *,
    encoding: Optional[str] = None,
    chunked: bool = False,
    method: str = "POST",
    path: str = "/",
) -> bytes:
    headers = [
        f"{method} {path} HTTP/1.1\r\n".encode(),
        b"Host: example.com\r\n",
    ]
    if encoding is not None:
        headers.append(f"Content-Encoding: {encoding}\r\n".encode())
    if chunked:
        headers.append(b"Transfer-Encoding: chunked\r\n")
        framed = frame_chunked(body)
    else:
        headers.append(f"Content-Length: {len(body)}\r\n".encode())
        framed = body
    headers.append(b"Connection: keep-alive\r\n\r\n")
    return b"".join(headers) + framed


def make_parser(
    loop: asyncio.AbstractEventLoop, limit: Optional[int]
) -> HttpRequestParser:
    return HttpRequestParser(
        mock.Mock(),
        loop,
        2**16,
        payload_exception=RequestPayloadError,
        decoded_body_limit=limit,
    )


def feed_until_message(
    raw: bytes, shard_size: int, limit: Optional[int]
) -> StreamReader:
    loop = asyncio.get_running_loop()
    parser = make_parser(loop, limit)
    payload: Optional[StreamReader] = None
    for start in range(0, len(raw), shard_size):
        messages, _, _ = parser.feed_data(raw[start : start + shard_size])
        for _, candidate in messages:
            payload = candidate
        assert payload is not None or not messages
        if payload is not None and isinstance(payload.exception(), PayloadSizeError):
            break
    assert payload is not None
    return payload


async def wait_for_predicate(
    predicate: Callable[[], bool], timeout: float = 2.0
) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


def make_mock_transport() -> mock.Mock:
    transport = mock.Mock(spec=asyncio.Transport)
    transport.is_closing.return_value = False
    transport.get_write_buffer_size.return_value = 0
    return transport


async def make_protocol(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    *,
    limit: Optional[int] = 4,
    transport: Optional[mock.Mock] = None,
) -> web.RequestHandler:
    server = web.Server(handler, decoded_body_limit=limit, lingering_time=0)
    protocol = server()
    protocol.connection_made(transport or make_mock_transport())
    return protocol


def transport_output(transport: mock.Mock) -> bytes:
    return b"".join(
        call.args[0]
        for call in transport.write.call_args_list
        if call.args and isinstance(call.args[0], (bytes, bytearray))
    )


async def test_application_handler_args_reach_request_handler() -> None:
    app = web.Application(handler_args={"decoded_body_limit": 1234})
    handler = app._make_handler()
    try:
        assert handler._kwargs["decoded_body_limit"] == 1234
    finally:
        await handler.shutdown()


async def test_invalid_decoded_body_limit() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response()

    server = web.Server(handler, decoded_body_limit=0)
    with pytest.raises(ValueError, match="positive integer"):
        server()


async def test_none_limit_keeps_stream_unbounded() -> None:
    loop = asyncio.get_running_loop()
    payload = StreamReader(mock.Mock(), 2**16, loop=loop, decoded_body_limit=None)
    payload.feed_data(b"x" * (3 * 1024 * 1024))
    assert payload.total_bytes == 3 * 1024 * 1024
    assert payload.decoded_body_remaining is None


@pytest.mark.parametrize("shard_size", [1, 2, 3, 5, 8, 64, 10_000])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    ("encoding", "compressed"),
    [("deflate", DEFLATE_DATA), ("gzip", gzip.compress(PLAIN_DATA))],
)
async def test_compressed_limit_is_stable_across_framing_and_shards(
    encoding: str,
    compressed: bytes,
    chunked: bool,
    shard_size: int,
) -> None:
    raw = raw_message(compressed, encoding=encoding, chunked=chunked)

    allowed = feed_until_message(raw, shard_size, len(PLAIN_DATA))
    assert await allowed.read() == PLAIN_DATA

    rejected = feed_until_message(raw, shard_size, len(PLAIN_DATA) - 1)
    exc = rejected.exception()
    assert isinstance(exc, PayloadSizeError)
    assert exc.limit == len(PLAIN_DATA) - 1
    with pytest.raises(PayloadSizeError):
        await rejected.read()


@pytest.mark.parametrize("shard_size", [1, 7, 65_537])
@pytest.mark.parametrize("encoding", [None, "gzip", "deflate"])
async def test_valid_bodies_keep_content_and_order(
    encoding: Optional[str], shard_size: int
) -> None:
    plain = (b"0123456789 abcdef " * 5000)[:200_003]
    if encoding == "gzip":
        body = gzip.compress(plain)
    elif encoding == "deflate":
        compressor = zlib.compressobj(wbits=15)
        body = compressor.compress(plain) + compressor.flush()
    else:
        body = plain

    raw = raw_message(body, encoding=encoding, chunked=True)
    payload = feed_until_message(raw, shard_size, 10**9)

    chunks = []
    while True:
        chunk = await payload.readany()
        if not chunk:
            break
        chunks.append(chunk)

    assert b"".join(chunks) == plain
    assert payload.total_bytes == len(plain)


async def test_consumed_bytes_keep_quota_and_read_apis_share_it() -> None:
    loop = asyncio.get_running_loop()
    protocol = mock.Mock()
    payload = StreamReader(
        protocol, 2**16, loop=loop, decoded_body_limit=4
    )

    payload.feed_data(b"ab")
    assert await payload.read(2) == b"ab"
    assert payload.total_bytes == 2

    # An empty current buffer must not make the decoder think the two consumed
    # bytes have given their quota back.
    with pytest.raises(PayloadSizeError) as exc_info:
        payload.feed_data(b"cde")
    assert exc_info.value.limit == 4
    assert exc_info.value.actual_size == 5
    assert payload.total_bytes == 2

    for read_call in (payload.read(), payload.readany(), payload.read(1)):
        with pytest.raises(PayloadSizeError):
            await read_call


async def test_request_read_translates_limit_to_413_exception() -> None:
    from aiohttp.test_utils import make_mocked_request

    loop = asyncio.get_running_loop()
    payload = StreamReader(mock.Mock(), 2**16, loop=loop, decoded_body_limit=4)
    with pytest.raises(PayloadSizeError):
        payload.feed_data(b"abcde")
    request = make_mocked_request("POST", "/", payload=payload)

    with pytest.raises(web.HTTPRequestEntityTooLarge):
        await request.read()


async def test_post_translates_multipart_stream_limit() -> None:
    from aiohttp.test_utils import make_mocked_request

    loop = asyncio.get_running_loop()
    payload = StreamReader(mock.Mock(), 2**16, loop=loop, decoded_body_limit=4)
    body = (
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="a"\r\n\r\n'
        b"abcdefgh\r\n"
        b"--boundary--\r\n"
    )
    with pytest.raises(PayloadSizeError):
        payload.feed_data(body)

    request = make_mocked_request(
        "POST",
        "/",
        headers={
            "Content-Type": "multipart/form-data; boundary=boundary",
        },
        payload=payload,
    )
    with pytest.raises(web.HTTPRequestEntityTooLarge):
        await request.post()


async def test_memory_peak_for_one_64mib_gzip_bomb_feed() -> None:
    bomb_plain = b"\0" * (64 * 1024 * 1024)
    compressed = gzip.compress(bomb_plain, compresslevel=9)
    assert len(compressed) < 256 * 1024
    raw = raw_message(compressed, encoding="gzip")

    async def handler(request: web.Request) -> web.Response:
        await request.read()
        return web.Response()

    transport = make_mock_transport()
    protocol = await make_protocol(handler, limit=LOWMEM_LIMIT, transport=transport)
    task = protocol._task_handler
    assert task is not None

    started_tracemalloc = not tracemalloc.is_tracing()
    if started_tracemalloc:
        tracemalloc.start()
    try:
        current_before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

        # One network read containing headers and the whole tiny compressed bomb.
        protocol.data_received(raw)

        _, peak_after = tracemalloc.get_traced_memory()
        extra_peak = peak_after - current_before
    finally:
        if started_tracemalloc:
            tracemalloc.stop()

    assert extra_peak <= 2 * 1024 * 1024

    payload_exc: Optional[BaseException] = None
    request = protocol._current_request
    if request is not None:
        payload_exc = request.content.exception()
    if payload_exc is None and protocol._messages:
        payload_exc = protocol._messages[0][1].exception()
    assert isinstance(payload_exc, PayloadSizeError)

    with pytest.raises(PayloadSizeError):
        raise payload_exc

    # Let the protocol emit its 413 and finish.
    await asyncio.wait_for(task, timeout=2)
    output = transport_output(transport)
    assert b"413 Request Entity Too Large" in output
    assert transport.close.called


async def test_same_limit_exception_is_shared_by_layers() -> None:
    woke: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()
    handled_request: list[web.Request] = []

    async def handler(request: web.Request) -> web.Response:
        handled_request.append(request)
        try:
            await request.content.read()
        except BaseException as exc:
            woke.set_result(exc)
            raise
        return web.Response()

    transport = make_mock_transport()
    protocol = await make_protocol(handler, limit=4, transport=transport)
    task = protocol._task_handler
    assert task is not None

    protocol.data_received(raw_message(b"abcde"))
    exc = await asyncio.wait_for(woke, timeout=2)
    await asyncio.wait_for(task, timeout=2)

    assert isinstance(exc, PayloadSizeError)
    assert protocol._payload_size_error is exc
    assert handled_request
    assert handled_request[0].content.exception() is exc


@pytest.mark.parametrize(
    "read_mode", ["request_read", "content_iter", "post"]
)
async def test_all_request_reading_paths_wake_and_return_413(
    read_mode: str,
) -> None:
    woke: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()

    async def handler(request: web.Request) -> web.Response:
        try:
            if read_mode == "request_read":
                await request.read()
            elif read_mode == "content_iter":
                async for _ in request.content:
                    pass
            else:
                await request.post()
            raise AssertionError("body limit was not raised")
        except BaseException as exc:
            if not woke.done():
                woke.set_result(exc)
            raise

    if read_mode == "post":
        raw = (
            b"POST / HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/x-www-form-urlencoded\r\n"
            b"Content-Length: 10\r\n\r\nabcdefghij"
        )
    else:
        raw = raw_message(b"abcdefghij", chunked=True)

    transport = make_mock_transport()
    protocol = await make_protocol(handler, limit=4, transport=transport)
    task = protocol._task_handler
    assert task is not None
    protocol.data_received(raw)

    exc = await asyncio.wait_for(woke, timeout=2)
    if read_mode == "request_read" or read_mode == "post":
        assert isinstance(exc, web.HTTPRequestEntityTooLarge)
    else:
        assert isinstance(exc, PayloadSizeError)

    await asyncio.wait_for(task, timeout=2)
    output = transport_output(transport)
    assert b"HTTP/1.1 413" in output
    assert b"Connection: close" in output
    assert protocol._close
    assert transport.close.called


def two_content_length_requests(first_body: bytes, second_body: bytes) -> bytes:
    return raw_message(first_body, path="/first") + raw_message(
        second_body, path="/second"
    )


async def test_pipelined_requests_have_independent_full_quotas() -> None:
    seen: list[bytes] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(await request.read())
        return web.Response(body=b"ok")

    transport = make_mock_transport()
    protocol = await make_protocol(handler, limit=4, transport=transport)
    protocol.data_received(two_content_length_requests(b"abcd", b"efgh"))

    await wait_for_predicate(lambda: len(seen) == 2)
    assert seen == [b"abcd", b"efgh"]

    output = transport_output(transport)
    assert output.count(b"HTTP/1.1 200 OK") == 2
    assert not protocol._close


async def test_request_after_oversize_pipeline_is_not_dispatched() -> None:
    calls: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        calls.append(request.path)
        await request.read()
        return web.Response()

    good = raw_message(b"abcd", path="/good")
    bad = raw_message(b"abcde", path="/bad")
    trailing = raw_message(b"efgh", path="/trailing")

    transport = make_mock_transport()
    protocol = await make_protocol(handler, limit=4, transport=transport)
    task = protocol._task_handler
    assert task is not None

    # Oversize request and another request already in the same segment.
    protocol.data_received(bad + trailing)
    await asyncio.wait_for(task, timeout=2)

    assert calls == ["/bad"]
    output = transport_output(transport)
    assert output.count(b"HTTP/1.1 413") == 1
    assert b"/trailing" not in output

    # Later segments must also be ignored rather than parsed as a new request.
    protocol.data_received(good)
    await asyncio.sleep(0)
    assert calls == ["/bad"]
    assert transport.close.called


async def test_queued_oversize_request_after_legal_request_gets_413() -> None:
    calls: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        calls.append(request.path)
        await request.read()
        return web.Response()

    legal = raw_message(b"abcd", path="/legal")
    bad = raw_message(b"abcde", path="/bad")
    trailing = raw_message(b"efgh", path="/trailing")

    transport = make_mock_transport()
    protocol = await make_protocol(handler, limit=4, transport=transport)
    task = protocol._task_handler
    assert task is not None

    protocol.data_received(legal + bad + trailing)
    await asyncio.wait_for(task, timeout=2)

    assert calls == ["/legal", "/bad"]
    output = transport_output(transport)
    assert output.count(b"HTTP/1.1 200 OK") == 1
    assert output.count(b"HTTP/1.1 413") == 1
    assert b"/trailing" not in output


async def test_handler_return_without_reading_still_gets_413() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(text="ignored")

    transport = make_mock_transport()
    protocol = await make_protocol(handler, limit=4, transport=transport)
    task = protocol._task_handler
    assert task is not None

    protocol.data_received(raw_message(b"abcde"))
    await asyncio.wait_for(task, timeout=2)

    output = transport_output(transport)
    assert output.count(b"HTTP/1.1 413") == 1
    assert b"ignored" not in output
    assert transport.close.called


async def test_headers_already_sent_are_closed_without_second_response() -> None:
    woke: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        try:
            async for _ in request.content:
                pass
        except BaseException as exc:
            woke.set_result(exc)
            raise
        return response

    transport = make_mock_transport()
    protocol = await make_protocol(handler, limit=4, transport=transport)
    task = protocol._task_handler
    assert task is not None

    protocol.data_received(raw_message(b"abcde"))
    exc = await asyncio.wait_for(woke, timeout=2)
    assert isinstance(exc, PayloadSizeError)

    # The outer protocol task treats a response already started as a client
    # disconnect/close condition and exits its loop normally.
    await asyncio.wait_for(task, timeout=2)
    assert task.exception() is None

    output = transport_output(transport)
    assert b"HTTP/1.1 200" in output
    assert output.count(b"HTTP/1.1") == 1
    assert b"413" not in output
    assert transport.close.called
