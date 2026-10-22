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
from pipecat.pipeline.worker import PipelineParams
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

    One response renders at a time and up to ``max_queued`` flushed utterances wait behind
    it; a flush past that is rejected. An utterance's response starts with its first text
    when nothing is rendering, before the flush that ends it. A connection's response n
    renders as ``_pcm(n)``, in ``chunks`` pieces spread over ``render_seconds``.
    """

    def __init__(
        self,
        *,
        init_error: dict | None = None,
        fail_first: dict | None = None,
        fail_first_open: dict | None = None,
        fail_open_times: int = 1,
        render_seconds: float = 0.0,
        chunks: int = 1,
        max_queued: int = 20,
        hold_first: bool = False,
        ack_cancels: bool = True,
    ):
        self.init_error = init_error  # sent instead of sessionCreated, then the socket closes
        self.fail_first = fail_first  # the first response fails with this error after its flush
        # A connection's first response fails with this error at its utterance's second text,
        # before the flush, on the first `fail_open_times` connections, after rendering
        # AUDIO_1. As on the server, the error is fatal and the connection closes.
        self.fail_first_open = fail_first_open
        self.fail_open_times = fail_open_times
        self.render_seconds = render_seconds
        self.chunks = chunks
        self.max_queued = max_queued
        self.hold_first = hold_first  # the first response starts only when a cancel arrives
        self.ack_cancels = ack_cancels
        self.paths: list[str] = []
        self.connections: list[list[dict]] = []  # the messages received, per connection

    @property
    def messages(self) -> list[dict]:
        return [m for messages in self.connections for m in messages]

    def of_type(self, type: str) -> list[dict]:
        return [m for m in self.messages if m["type"] == type]

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        messages = []
        self.connections.append(messages)
        count = 0
        current = None  # {"rid", "flushed", "created"}: the response starting or rendering
        queued = 0  # flushed utterances waiting behind `current`
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
            nonlocal current, queued, open_text
            rid = current["rid"]
            if rid == "resp_1" and self.fail_first:
                await asyncio.sleep(self.render_seconds)
                await send(
                    {"type": "error", "fatal": False, "response_id": rid, "error": self.fail_first}
                )
                await send({"type": "responseDone", "response_id": rid, "status": "cancelled"})
            else:
                pcm = _pcm(int(rid.split("_")[1]))
                size = len(pcm) // self.chunks
                for i in range(self.chunks):
                    await asyncio.sleep(self.render_seconds / self.chunks)
                    await _audio(ws, rid, pcm[i * size : (i + 1) * size])
                await send({"type": "responseDone", "response_id": rid, "status": "completed"})
            # Like the server, start the next utterance only after closing the finished one.
            await asyncio.sleep(0.05)
            current = None
            if queued:
                queued -= 1
                await start(flushed=True)
            elif open_text:
                await start(flushed=False)

        try:
            async for raw in ws:
                msg = json.loads(raw)
                messages.append(msg)
                if msg["type"] == "initializeConnection":
                    if self.init_error:
                        await send({"type": "error", "fatal": True, "error": self.init_error})
                        await ws.close(4401)
                        return
                    await send({"type": "sessionCreated", "session_id": "s", "sample_rate": 24000})
                elif msg["type"] == "sendText":
                    if len(open_text + msg.get("text", "")) > 8000:
                        error = {"type": "invalid_request", "message": "over 8000 characters"}
                        await send({"type": "error", "fatal": False, "error": error})
                        continue
                    if (
                        self.fail_first_open
                        and len(self.connections) <= self.fail_open_times
                        and current
                        and current["rid"] == "resp_1"
                        and not current["flushed"]
                        and not msg.get("flush")
                    ):
                        await _audio(ws, "resp_1", AUDIO_1)
                        error = {"type": "error", "fatal": True, "response_id": "resp_1"}
                        await send({**error, "error": self.fail_first_open})
                        await ws.close(1011, "utterance failed")
                        return
                    open_text += msg.get("text", "")
                    if not msg.get("flush"):
                        if open_text and not current and not queued:
                            await start(flushed=False)
                    elif current and not current["flushed"]:
                        current["flushed"] = True  # the open utterance's own response
                        open_text = ""
                        if current["created"]:
                            render = asyncio.create_task(finish())
                    elif not current:
                        open_text = ""
                        await start(flushed=True)
                    elif queued < self.max_queued:
                        open_text, queued = "", queued + 1
                    else:
                        error = {"type": "invalid_request", "message": "queue full"}
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
                    current, queued, open_text = None, 0, ""
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


async def _run_ws(fake: _FakeKalpa, frames: list, pipeline_params=None, **kwargs):
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = KalpaTTSService(api_key="test-key", base_url=f"ws://{host}:{port}", **kwargs)
        down, up = await asyncio.wait_for(
            run_test(tts, frames_to_send=frames, pipeline_params=pipeline_params), timeout=10
        )
    return tts, down, up


@pytest.mark.asyncio
async def test_kalpa_tts_streams_a_reply_over_one_utterance():
    """LLM tokens go out verbatim and the reply's audio stays 24 kHz in a 16 kHz pipeline."""
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
        pipeline_params=PipelineParams(audio_out_sample_rate=16000),
    )

    assert not _errors(down + up)
    assert fake.paths == [f"/v1/tts/{DEFAULT_VOICE_ID}/stream"]
    init = fake.of_type("initializeConnection")[0]
    assert init["api_key"] == "test-key"
    assert init["model"] == "kalpa-tts-1"
    assert init["generation_config"] == {}
    assert "sample_rate" not in init and "params" not in init
    sends = fake.of_type("sendText")
    assert [m.get("text") for m in sends if not m.get("flush")] == ["Hello", " there."]
    assert sum(1 for m in sends if m.get("flush")) == 1

    assert _audio_of(down) == _pcm(1)
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert all(f.sample_rate == 24000 for f in audio)
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
    fake = _FakeKalpa(render_seconds=1.5, ack_cancels=False)
    _, down, up = await _run_ws(
        fake,
        [
            *_reply("Let me tell you a long story."),
            SleepFrame(sleep=0.2),
            InterruptionFrame(),
            SleepFrame(sleep=0.1),
            *_reply("Sure."),
            SleepFrame(sleep=3),
        ],
    )

    assert len(fake.paths) == 2
    assert _audio_of(down) == _pcm(1)  # the new connection's first response


@pytest.mark.asyncio
async def test_kalpa_tts_speech_inside_a_reply_keeps_every_part():
    """Speech inserted mid-reply reaches Kalpa after the reply, in the order it all plays.

    Rendered first, the speech would leave the playing reply without audio for longer than its
    context's timeout, and the rest of the reply would be lost.
    """
    fake = _FakeKalpa(render_seconds=0.8, chunks=8)
    _, down, up = await _run_ws(
        fake,
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("First part of the reply."),
            SleepFrame(sleep=0.05),
            TTSSpeakFrame("One moment."),
            TTSSpeakFrame("Still checking."),
            LLMTextFrame(" And the rest."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=3.5),
        ],
        stop_frame_timeout_s=0.5,
    )

    assert not _errors(down + up)
    sends = ["flush" if m.get("flush") else m["text"] for m in fake.of_type("sendText")]
    assert sends == [
        "First part of the reply.",
        " And the rest.",
        "flush",
        "One moment.",
        "flush",
        "Still checking.",
        "flush",
    ]
    started = [f.context_id for f in down if isinstance(f, TTSStartedFrame)]
    stopped = [f.context_id for f in down if isinstance(f, TTSStoppedFrame)]
    assert len(started) == 3 and stopped == started
    assert [_audio_of(down, context_id) for context_id in started] == [_pcm(1), _pcm(2), _pcm(3)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,category,rest,resent,audio",
    [
        # The held speech and the rest of the reply go out on the new connection.
        (
            "overloaded",
            ErrorCategory.SERVER,
            [SleepFrame(sleep=0.2), LLMTextFrame(" Third part.")],
            ["One moment.", None, " Third part.", None, "Reply B.", None],
            [AUDIO_1, _pcm(1), _pcm(2), _pcm(3)],
        ),
        # The reply's flush, and the speech it releases, are on their way when the failure
        # arrives; Kalpa drops them, so the speech goes out again on the new connection.
        (
            "context_length_exceeded",
            ErrorCategory.APPLICATION,
            [],
            ["One moment.", None, "Reply B.", None],
            [AUDIO_1, _pcm(1), _pcm(2)],
        ),
    ],
    ids=["rest-of-reply", "flush-in-flight"],
)
async def test_kalpa_tts_reply_that_fails_before_its_flush_replaces_the_connection(
    error_type, category, rest, resent, audio
):
    """Kalpa ends the connection on a reply that fails before its flush.

    What it rendered still plays, speech held behind it and the text after the failure go out
    on a new connection, and every context gets only its own audio.
    """
    fake = _FakeKalpa(fail_first_open={"type": error_type, "message": "Failed."})
    tts, down, up = await _run_ws(
        fake,
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("First part of reply A."),
            SleepFrame(sleep=0.05),
            TTSSpeakFrame("One moment."),
            LLMTextFrame(" Second part."),  # Kalpa fails the reply here
            *rest,
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.3),
            *_reply("Reply B."),
            SleepFrame(sleep=0.3),
        ],
    )

    assert [e.category for e in _errors(down + up)] == [category]
    assert tts.is_usable and len(fake.connections) == 2
    assert [m.get("text") for m in fake.connections[1] if m["type"] == "sendText"] == resent
    started = [f.context_id for f in down if isinstance(f, TTSStartedFrame)]
    assert [_audio_of(down, context_id) for context_id in started] == audio


@pytest.mark.asyncio
async def test_kalpa_tts_reply_that_goes_quiet_does_not_hold_later_speech():
    """A reply whose context times out before the reply ends is ended then.

    The base class doesn't end a context that is no longer playing, so the reply's utterance
    would otherwise stay open and hold the speech behind it, and every later reply.
    """
    fake = _FakeKalpa()
    _, down, up = await _run_ws(
        fake,
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Let me check."),
            SleepFrame(sleep=0.05),
            TTSSpeakFrame("One moment."),
            SleepFrame(sleep=1),  # past the reply context's timeout
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.3),
            *_reply("Reply B."),
            SleepFrame(sleep=0.5),
        ],
        stop_frame_timeout_s=0.5,
    )

    sends = ["flush" if m.get("flush") else m["text"] for m in fake.of_type("sendText")]
    assert sends == ["Let me check.", "flush", "One moment.", "flush", "Reply B.", "flush"]
    _, speech, reply_b = [f.context_id for f in down if isinstance(f, TTSStartedFrame)]
    assert _audio_of(down, speech) == _pcm(2)
    assert _audio_of(down, reply_b) == _pcm(3)
    stopped = [f.context_id for f in down if isinstance(f, TTSStoppedFrame)]
    assert speech in stopped and reply_b in stopped


@pytest.mark.asyncio
async def test_kalpa_tts_failed_replies_do_not_count_as_failing_connections():
    """Kalpa closing the connection on failed replies doesn't make the service give up on it."""
    overloaded = {"type": "overloaded", "message": "The model is temporarily overloaded."}
    fake = _FakeKalpa(fail_first_open=overloaded, fail_open_times=3)
    failing = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("First part."),
        SleepFrame(sleep=0.05),
        LLMTextFrame(" Second part."),
        LLMFullResponseEndFrame(),
        SleepFrame(sleep=0.2),
    ]
    tts, down, up = await _run_ws(fake, [*failing * 3, *_reply("Hello."), SleepFrame(sleep=0.3)])

    assert [e.category for e in _errors(down + up)] == [ErrorCategory.SERVER] * 3
    assert tts.is_usable and len(fake.connections) == 4
    assert _audio_of(down) == AUDIO_1 * 3 + _pcm(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_ended", [False, True], ids=["held", "queued-on-kalpa"])
async def test_kalpa_tts_uninterruptible_speech_survives_a_barge_in(reply_ended):
    """A barge-in cancels the reply, and uninterruptible speech behind it still plays.

    Whether it was held behind the reply or already queued on Kalpa, where the cancel drops it,
    the speech goes out after the cancel and before the next reply.
    """
    fake = _FakeKalpa(render_seconds=0.5)
    speech = TTSSpeakFrame("Please hold.")
    speech.interruptible = False
    _, down, up = await _run_ws(
        fake,
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("First part of the reply."),
            SleepFrame(sleep=0.05),
            speech,
            *([LLMFullResponseEndFrame()] if reply_ended else []),
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
            *_reply("Sure."),
            SleepFrame(sleep=2.5),
        ],
    )

    assert not _errors(down + up)
    assert len(fake.paths) == 1
    types = [m["type"] for m in fake.messages]
    assert types.count("cancelResponse") == 1
    after = fake.messages[types.index("cancelResponse") + 1 :]
    assert ["flush" if m.get("flush") else m["text"] for m in after] == [
        "Please hold.",
        "flush",
        "Sure.",
        "flush",
    ]
    reply, held, reply_b = [f.context_id for f in down if isinstance(f, TTSStartedFrame)]
    assert _audio_of(down, reply) == b""
    assert _audio_of(down, held) == _pcm(2)
    assert _audio_of(down, reply_b) == _pcm(3)


@pytest.mark.asyncio
async def test_kalpa_tts_text_over_the_utterance_limit_is_split(monkeypatch):
    """Text longer than one utterance takes is spoken as several, cut at a space.

    Each flush waits for room in Kalpa's queue, here one utterance deep.
    """
    monkeypatch.setattr("pipecat.services.kalpa.tts._MAX_QUEUED_FLUSHES", 1)
    fake = _FakeKalpa(render_seconds=0.2, max_queued=1)
    text = "word " * 3300  # 16,500 characters
    _, down, up = await _run_ws(fake, [TTSSpeakFrame(text), SleepFrame(sleep=1)])

    assert not _errors(down + up)
    sent = [m["text"] for m in fake.of_type("sendText") if not m.get("flush")]
    assert "".join(sent) == text
    assert [len(t) for t in sent] == [8000, 8000, 500]
    assert _audio_of(down) == _pcm(1) + _pcm(2) + _pcm(3)


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
@pytest.mark.parametrize(
    "error_type,category",
    [
        ("allowance_exhausted", ErrorCategory.QUOTA),
        ("invalid_request", ErrorCategory.INVALID_REQUEST),  # e.g. an unknown model
    ],
)
async def test_kalpa_tts_fatal_session_error_makes_the_service_unusable(error_type, category):
    """A fatal error while the session opens is reported and not retried."""
    fake = _FakeKalpa(init_error={"type": error_type, "message": "Rejected."})
    tts, down, up = await _run_ws(fake, [*_reply("Hello."), SleepFrame(sleep=0.2)])

    errors = _errors(down + up)
    assert errors and errors[0].category == category
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
        api_key="test-key",
        base_url=str(client.make_url("")).rstrip("/"),
        stop_frame_timeout_s=0.5,
        **kwargs,
    )
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hello from Kalpa.")])
    return tts, down, up


@pytest.mark.asyncio
async def test_kalpa_http_tts_streams_the_raw_pcm_body(aiohttp_client):
    """The sentence is posted and the body plays as it arrives."""
    requests = []

    async def handler(request):
        requests.append((request.match_info["voice_id"], request.headers, await request.json()))
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(AUDIO_1[:-1])  # chunks may split a sample
        await response.write(AUDIO_1[-1:] + AUDIO_2)
        await response.write_eof()
        return response

    _, down, up = await _run_http(aiohttp_client, handler)

    assert not _errors(down + up)
    voice_id, headers, body = requests[0]
    assert voice_id == DEFAULT_VOICE_ID
    assert headers["Authorization"] == "Bearer test-key"
    assert body == {"text": "Hello from Kalpa.", "model": "kalpa-tts-1"}
    assert _audio_of(down) == AUDIO_1 + AUDIO_2


@pytest.mark.asyncio
async def test_kalpa_http_tts_text_over_the_request_limit_is_split(aiohttp_client):
    """Text longer than one request takes goes out as several requests, cut at a space."""
    texts = []

    async def handler(request):
        texts.append((await request.json())["text"])
        return web.Response(body=AUDIO_1, content_type="audio/pcm")

    app = web.Application()
    app.router.add_post("/v1/tts/{voice_id}/stream", handler)
    client = await aiohttp_client(app)
    tts = KalpaHttpTTSService(api_key="test-key", base_url=str(client.make_url("")).rstrip("/"))
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="word " * 1700)])

    assert not _errors(down + up)
    assert "".join(texts) == "word " * 1700
    assert [len(t) for t in texts] == [8000, 500]
    assert _audio_of(down) == AUDIO_1 + AUDIO_1


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

    tts, down, up = await _run_http(aiohttp_client, handler)

    errors = _errors(down + up)
    assert errors and errors[0].category == category
    assert tts.is_usable == usable
