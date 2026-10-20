#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for KalpaTTSService and KalpaHttpTTSService."""

import asyncio
import base64
import json
import time

import pytest
import websockets
from aiohttp import web
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.services.kalpa.tts import KalpaHttpTTSService, KalpaTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory

DEFAULT_VOICE_ID = "18c26f49-9f9c-400c-bec9-7467f28e1db5"
OTHER_VOICE_ID = "dda5319f-3b81-4124-b69e-9241eb510b77"

AUDIO_1 = b"\x01\x00" * 480
AUDIO_2 = b"\x02\x00" * 480
STALE_AUDIO = b"\x7f\x00" * 480


def _pcm(response: int) -> bytes:
    """The audio the fake renders for its n-th response."""
    return bytes([response, 0]) * 480


def _reply(text: str) -> list:
    return [LLMFullResponseStartFrame(), LLMTextFrame(text), LLMFullResponseEndFrame()]


def _audio_of(frames, context_id: str | None = None) -> bytes:
    return b"".join(
        f.audio
        for f in frames
        if isinstance(f, TTSAudioRawFrame) and context_id in (None, f.context_id)
    )


def _errors(frames) -> list[ErrorFrame]:
    return [f for f in frames if isinstance(f, ErrorFrame)]


class _FakeKalpa:
    """An in-process ``WSS /v1/tts/{voice_id}/stream`` with the server's queueing rules.

    One response renders at a time and one flushed utterance may wait behind it; a third
    flush is rejected. An utterance's response starts with its first text when nothing is
    rendering, before the flush that ends it. Response n renders as ``_pcm(n)``.
    """

    def __init__(
        self,
        *,
        init_error: dict | None = None,
        fail_first: dict | None = None,
        render_seconds: float = 0.0,
        hold_first: bool = False,
        ack_cancels: bool = True,
    ):
        self.init_error = init_error  # sent instead of sessionCreated, then the socket closes
        self.fail_first = fail_first  # the first response fails with this error
        self.render_seconds = render_seconds
        self.hold_first = hold_first  # the first response starts only when a cancel arrives
        self.ack_cancels = ack_cancels
        self.paths: list[str] = []
        self.messages: list[dict] = []

    def of_type(self, type: str) -> list[dict]:
        return [m for m in self.messages if m["type"] == type]

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        count = 0
        current = None  # {"rid", "flushed", "created"}: the response starting or rendering
        pending = False  # a flushed utterance waits behind `current`
        open_text = ""  # buffered text of the utterance not flushed yet
        render = None

        async def send(msg: dict):
            await ws.send(json.dumps(msg))

        async def start(flushed: bool):
            nonlocal count, current, render
            count += 1
            current = {"rid": f"resp_{count}", "flushed": flushed, "created": False}
            if count == 1 and self.hold_first:
                return
            current["created"] = True
            await send({"type": "responseCreated", "response_id": current["rid"]})
            if flushed:
                render = asyncio.create_task(finish())

        async def finish():
            nonlocal current, pending, open_text
            rid = current["rid"]
            await asyncio.sleep(self.render_seconds)
            if rid == "resp_1" and self.fail_first:
                await send(
                    {"type": "error", "fatal": False, "response_id": rid, "error": self.fail_first}
                )
                await send({"type": "responseDone", "response_id": rid, "status": "cancelled"})
            else:
                await _audio(ws, rid, _pcm(int(rid.split("_")[1])))
                await send({"type": "responseDone", "response_id": rid, "status": "completed"})
            # Like the server, free the slot only after closing the finished response.
            await asyncio.sleep(0.05)
            current = None
            if pending:
                pending = False
                await start(flushed=True)
            elif open_text:
                await start(flushed=False)

        try:
            async for raw in ws:
                msg = json.loads(raw)
                self.messages.append(msg)
                if msg["type"] == "initializeConnection":
                    if self.init_error:
                        await send({"type": "error", "fatal": True, "error": self.init_error})
                        await ws.close(4401)
                        return
                    rate = msg["sample_rate"]
                    await send({"type": "sessionCreated", "session_id": "s", "sample_rate": rate})
                elif msg["type"] == "sendText":
                    open_text += msg.get("text", "")
                    if not msg.get("flush"):
                        if open_text and not current and not pending:
                            await start(flushed=False)
                    elif current and not current["flushed"]:
                        current["flushed"] = True  # the open utterance's own response
                        open_text = ""
                        if current["created"]:
                            render = asyncio.create_task(finish())
                    elif not current:
                        open_text = ""
                        await start(flushed=True)
                    elif not pending:
                        open_text, pending = "", True
                    else:
                        error = {"type": "invalid_request", "message": "already queued"}
                        await send({"type": "error", "fatal": False, "error": error})
                elif msg["type"] == "cancelResponse":
                    # Let the client's next reply reach the socket first.
                    await asyncio.sleep(0.1)
                    if render:
                        render.cancel()
                    if current:
                        rid = current["rid"]
                        if not current["created"]:
                            await send({"type": "responseCreated", "response_id": rid})
                            await _audio(ws, rid, STALE_AUDIO)
                        await send(
                            {"type": "responseDone", "response_id": rid, "status": "cancelled"}
                        )
                    current, pending, open_text = None, False, ""
                    if self.ack_cancels:
                        await send({"type": "cancelled"})
        except websockets.ConnectionClosed:
            pass
        finally:
            if render:
                render.cancel()


async def _audio(ws, rid: str, pcm: bytes):
    pcm_b64 = base64.b64encode(pcm).decode()
    await ws.send(json.dumps({"type": "responseAudio", "response_id": rid, "pcm_b64": pcm_b64}))


async def _run_ws(fake: _FakeKalpa, frames: list, **kwargs):
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = KalpaTTSService(api_key="test-key", base_url=f"ws://{host}:{port}", **kwargs)
        down, up = await asyncio.wait_for(run_test(tts, frames_to_send=frames), timeout=10)
    return tts, down, up


@pytest.mark.asyncio
async def test_kalpa_tts_streams_a_reply_over_one_utterance():
    """LLM tokens go out verbatim and the reply's flush returns its audio at the session rate."""
    fake = _FakeKalpa()
    start = time.monotonic()
    _, down, up = await _run_ws(
        fake,
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello"),
            LLMTextFrame(" there."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.3),
        ],
        sample_rate=16000,
    )

    assert not _errors(down + up)
    assert fake.paths == [f"/v1/tts/{DEFAULT_VOICE_ID}/stream"]
    init = fake.of_type("initializeConnection")[0]
    assert init["api_key"] == "test-key"
    assert init["model"] == "kalpa-tts-1"
    assert init["generation_config"] == {}
    assert init["sample_rate"] == 16000
    assert "params" not in init
    sends = fake.of_type("sendText")
    assert [m.get("text") for m in sends if not m.get("flush")] == ["Hello", " there."]
    assert sum(1 for m in sends if m.get("flush")) == 1

    assert _audio_of(down) == _pcm(1)
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert all(f.sample_rate == 16000 for f in audio)
    types = [type(f) for f in down]
    assert types.index(TTSStartedFrame) < types.index(TTSAudioRawFrame)
    assert types.index(TTSStoppedFrame) > types.index(TTSAudioRawFrame)
    # The reply ends on `responseDone`, not after the base class's idle timeout.
    assert time.monotonic() - start < 2.5


@pytest.mark.asyncio
async def test_kalpa_tts_interruption_cancels_on_the_same_connection():
    """A barge-in cancels the reply, drops what it still sends, and the next reply plays."""
    fake = _FakeKalpa(hold_first=True)
    _, down, up = await _run_ws(
        fake,
        [
            *_reply("Let me tell you a long story."),
            SleepFrame(sleep=0.2),
            InterruptionFrame(),
            *_reply("Sure, go ahead."),
            SleepFrame(sleep=0.5),
        ],
    )

    assert not _errors(down + up)
    assert len(fake.paths) == 1
    assert len(fake.of_type("cancelResponse")) == 1
    assert _audio_of(down) == _pcm(2)


@pytest.mark.asyncio
async def test_kalpa_tts_unacknowledged_cancel_replaces_the_connection():
    """Without a `cancelled`, the next reply goes out on a new connection."""
    fake = _FakeKalpa(render_seconds=5, ack_cancels=False)
    _, down, up = await _run_ws(
        fake,
        [
            *_reply("Let me tell you a long story."),
            SleepFrame(sleep=0.2),
            InterruptionFrame(),
            SleepFrame(sleep=0.1),
            *_reply("Sure."),
            SleepFrame(sleep=6),
        ],
    )

    assert len(fake.paths) == 2
    assert _audio_of(down) == _pcm(1)  # the new connection's first response


@pytest.mark.asyncio
async def test_kalpa_tts_speech_inside_a_reply_keeps_every_part():
    """A TTSSpeakFrame mid-reply splits the reply in two on the server; all three parts play."""
    fake = _FakeKalpa(render_seconds=0.3)
    _, down, up = await _run_ws(
        fake,
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("First part of the reply."),
            SleepFrame(sleep=0.05),
            TTSSpeakFrame("One moment."),
            LLMTextFrame(" And the rest."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=1.5),
        ],
    )

    assert not _errors(down + up)
    started = [f.context_id for f in down if isinstance(f, TTSStartedFrame)]
    stopped = [f.context_id for f in down if isinstance(f, TTSStoppedFrame)]
    assert len(started) == 2 and stopped == started
    reply, speech = started
    assert _audio_of(down, reply) == _pcm(1) + _pcm(3)
    assert _audio_of(down, speech) == _pcm(2)


@pytest.mark.asyncio
async def test_kalpa_tts_pipeline_cancel_closes_the_connection():
    """Cancelling the pipeline tears the service down without an error."""
    fake = _FakeKalpa()
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = KalpaTTSService(api_key="test-key", base_url=f"ws://{host}:{port}")
        down, up = await asyncio.wait_for(
            run_test(
                tts,
                frames_to_send=[*_reply("Hello."), SleepFrame(sleep=0.3), CancelFrame()],
                send_end_frame=False,
            ),
            timeout=10,
        )

    assert not _errors(down + up)
    assert tts._websocket is None


@pytest.mark.asyncio
async def test_kalpa_tts_fatal_session_error_makes_the_service_unusable():
    """A fatal error while the session opens is reported and not retried."""
    fake = _FakeKalpa(init_error={"type": "allowance_exhausted", "message": "Allowance used up."})
    tts, down, up = await _run_ws(fake, [*_reply("Hello."), SleepFrame(sleep=0.2)])

    errors = _errors(down + up)
    assert errors and errors[0].category == ErrorCategory.QUOTA
    assert not tts.is_usable
    assert len(fake.paths) == 1


@pytest.mark.asyncio
async def test_kalpa_tts_failed_reply_keeps_the_service_usable():
    """A reply the server fails is reported, and the next reply plays on the same connection."""
    overloaded = {"type": "overloaded", "message": "The model is temporarily overloaded."}
    fake = _FakeKalpa(fail_first=overloaded)
    tts, down, up = await _run_ws(
        fake,
        [*_reply("Hello."), SleepFrame(sleep=0.3), *_reply("Hello again."), SleepFrame(sleep=0.3)],
    )

    errors = _errors(down + up)
    assert errors and errors[0].category == ErrorCategory.SERVER
    assert tts.is_usable
    assert len(fake.paths) == 1
    assert _audio_of(down) == _pcm(2)


@pytest.mark.asyncio
async def test_kalpa_tts_voice_change_reconnects_with_the_new_voice():
    """The voice rides the URL, so a voice update opens a new connection for it."""
    fake = _FakeKalpa()
    await _run_ws(
        fake,
        [
            TTSUpdateSettingsFrame(delta=KalpaTTSService.Settings(voice=OTHER_VOICE_ID)),
            SleepFrame(sleep=0.2),
            *_reply("Hello."),
            SleepFrame(sleep=0.3),
        ],
    )

    assert fake.paths == [
        f"/v1/tts/{DEFAULT_VOICE_ID}/stream",
        f"/v1/tts/{OTHER_VOICE_ID}/stream",
    ]


async def _run_http(aiohttp_client, handler, **kwargs):
    app = web.Application()
    app.router.add_post("/v1/tts/{voice_id}/stream", handler)
    client = await aiohttp_client(app)
    tts = KalpaHttpTTSService(
        api_key="test-key", base_url=str(client.make_url("")).rstrip("/"), **kwargs
    )
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hello from Kalpa.")])
    return tts, down, up


@pytest.mark.asyncio
async def test_kalpa_http_tts_streams_the_raw_pcm_body(aiohttp_client):
    """The sentence is posted with the pipeline's rate and the body plays as it arrives."""
    requests = []

    async def handler(request):
        requests.append((request.match_info["voice_id"], request.headers, await request.json()))
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(AUDIO_1[:-1])  # chunks may split a sample
        await response.write(AUDIO_1[-1:] + AUDIO_2)
        await response.write_eof()
        return response

    _, down, up = await _run_http(aiohttp_client, handler, sample_rate=8000)

    assert not _errors(down + up)
    voice_id, headers, body = requests[0]
    assert voice_id == DEFAULT_VOICE_ID
    assert headers["Authorization"] == "Bearer test-key"
    assert body == {
        "text": "Hello from Kalpa.",
        "model": "kalpa-tts-1",
        "output_format": "pcm_s16le",
        "sample_rate": 8000,
    }
    assert _audio_of(down) == AUDIO_1 + AUDIO_2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error_type,category,usable",
    [
        (400, "invalid_request", ErrorCategory.INVALID_REQUEST, False),  # unknown model
        (400, "context_length_exceeded", ErrorCategory.APPLICATION, True),
        (402, "allowance_exhausted", ErrorCategory.QUOTA, False),
        (503, "overloaded", ErrorCategory.SERVER, True),
    ],
)
async def test_kalpa_http_tts_errors_are_categorized(
    aiohttp_client, status, error_type, category, usable
):
    """Only errors that will recur cost the service its usability."""

    async def handler(request):
        error = {"type": error_type, "message": "Nope.", "request_id": "req_1"}
        return web.json_response({"error": error}, status=status)

    tts, down, up = await _run_http(aiohttp_client, handler, sample_rate=24000)

    errors = _errors(down + up)
    assert errors and errors[0].category == category
    assert tts.is_usable == usable
