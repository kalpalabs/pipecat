#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kalpa text-to-speech service implementations.

See https://docs.kalpalabs.ai/text-to-speech for the streaming WebSocket
(``WSS /v1/tts/{voice_id}/stream``) and the streaming HTTP endpoint
(``POST /v1/tts/{voice_id}/stream``) these services use.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import aiohttp
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService, WebsocketTTSService
from pipecat.utils.errors import ErrorCategory, classify_http_status_code
from pipecat.utils.tracing.service_decorators import traced_tts

_DEFAULT_VOICE_ID = "18c26f49-9f9c-400c-bec9-7467f28e1db5"  # Arjun
_DEFAULT_MODEL = "kalpa-tts-1"

# Rates Kalpa renders directly. Any other pipeline rate is served at 24 kHz, the model's own.
_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
_DEFAULT_SAMPLE_RATE = 24000

_SESSION_TIMEOUT_SECONDS = 10.0

# Kalpa closes a connection after 300 seconds without a client message.
_KEEPALIVE_INTERVAL_SECONDS = 60.0

# Kalpa ends every reply with `responseDone`, so the base class's idle timeout on an
# audio context is only a safety net. It has to outlast a slow first audio.
_STOP_FRAME_TIMEOUT_SECONDS = 15.0

# A cancel Kalpa doesn't acknowledge in time leaves the connection unusable.
_CANCEL_ACK_TIMEOUT_SECONDS = 1.0

# Kalpa's limit on one utterance's text.
_MAX_UTTERANCE_CHARS = 8000

# How long a flush waits for Kalpa to start the utterance queued before it.
_FLUSH_WAIT_SECONDS = 30.0

# Categories of the error types at https://docs.kalpalabs.ai/rate-limits-and-errors that are not
# the server failing. Rejected text fails only its own request, so it doesn't cost the service
# its usability.
_ERROR_CATEGORIES = {
    "authentication_error": ErrorCategory.AUTHENTICATION,
    "account_suspended": ErrorCategory.AUTHORIZATION,
    "not_found": ErrorCategory.INVALID_REQUEST,
    "allowance_exhausted": ErrorCategory.QUOTA,
    "rate_limit_exceeded": ErrorCategory.RATE_LIMIT,
    "invalid_request": ErrorCategory.APPLICATION,
    "context_length_exceeded": ErrorCategory.APPLICATION,
}


def _error_category(error_type: str | None, status: int | None = None) -> ErrorCategory:
    if error_type == "invalid_request" and status == 400:
        return ErrorCategory.INVALID_REQUEST  # an unknown model; rejected text is a 422
    category = _ERROR_CATEGORIES.get(error_type or "")
    if category:
        return category
    return classify_http_status_code(status) if status else ErrorCategory.SERVER


def _resolve_sample_rate(sample_rate: int) -> int:
    """Pick the rate Kalpa will synthesize at.

    Args:
        sample_rate: The pipeline's output rate.

    Returns:
        The pipeline's rate when Kalpa renders it, else 24000.
    """
    if sample_rate in _SAMPLE_RATES:
        return sample_rate
    logger.warning(
        f"Kalpa cannot render {sample_rate} Hz (supports {list(_SAMPLE_RATES)}); "
        f"synthesizing at {_DEFAULT_SAMPLE_RATE} Hz and resampling"
    )
    return _DEFAULT_SAMPLE_RATE


@dataclass
class KalpaTTSSettings(TTSSettings):
    """Settings for KalpaTTSService and KalpaHttpTTSService.

    ``voice`` is a voice id from ``GET /v1/voices`` and ``model`` a model id
    from ``GET /v1/models``. ``language`` is not sent to Kalpa, which reads the
    language from the text; it only guides sentence splitting.
    """


def _default_settings(settings: KalpaTTSSettings | None) -> KalpaTTSSettings:
    defaults = KalpaTTSSettings(model=_DEFAULT_MODEL, voice=_DEFAULT_VOICE_ID, language=None)
    if settings is not None:
        defaults.apply_update(settings)
    return defaults


@dataclass(eq=False)
class _Utterance:
    """Text sent to Kalpa for one context, from its first text until its `responseDone`."""

    context_id: str
    chars: int = 0
    flushed: bool = False
    # Named by `responseCreated`, which Kalpa sends for utterances in the order they arrived.
    response_id: str | None = None


class KalpaTTSService(WebsocketTTSService):
    """Kalpa WebSocket text-to-speech service.

    Streams speech over one connection to ``WSS /v1/tts/{voice_id}/stream`` for
    the whole conversation. Each reply is one utterance on the server, which
    starts speaking it as soon as it holds enough text ending in a complete
    sentence, and ends it on the reply's flush. LLM tokens are therefore
    forwarded as they arrive by default; pass
    ``text_aggregation_mode=TextAggregationMode.SENTENCE`` to aggregate
    sentences before sending instead.

    Interruptions cancel the reply on the server and keep the connection. The
    cancel covers everything the server holds for the connection, including
    uninterruptible speech queued behind the interrupted reply.

    Event handlers:

    - on_connected: Called when the websocket connection is established.
    - on_disconnected: Called when the websocket connection is closed.
    - on_connection_error: Called when a websocket connection error occurs.

    Example::

        tts = KalpaTTSService(
            api_key=os.getenv("KALPA_API_KEY"),
            settings=KalpaTTSService.Settings(
                voice="18c26f49-9f9c-400c-bec9-7467f28e1db5",  # Arjun
            ),
        )
    """

    Settings = KalpaTTSSettings
    _settings: Settings
    # The rate Kalpa synthesizes at, resolved in setup() from the pipeline's.
    _kalpa_sample_rate: int

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.kalpalabs.ai",
        sample_rate: int | None = None,
        text_aggregation_mode: TextAggregationMode = TextAggregationMode.TOKEN,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Kalpa WebSocket TTS service.

        Args:
            api_key: Kalpa API key for authentication.
            base_url: WebSocket base URL for the Kalpa API. Defaults to
                ``wss://api.kalpalabs.ai``.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline
                default. A rate Kalpa does not render is replaced with 24000
                and resampled by the output transport.
            text_aggregation_mode: How to aggregate incoming text before sending.
                Defaults to ``TextAggregationMode.TOKEN``, streaming LLM tokens
                straight to Kalpa, which picks its own sentence boundaries.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to ``WebsocketTTSService``.
        """
        if not api_key:
            raise ValueError("Kalpa API key is required")
        kwargs.setdefault("stop_frame_timeout_s", _STOP_FRAME_TIMEOUT_SECONDS)
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=False,
            # One reply generates at a time, so the next waits for the current one.
            pause_frame_processing=True,
            text_aggregation_mode=text_aggregation_mode,
            # Kalpa appends each `sendText` verbatim, so consecutive sentences would
            # otherwise glue together. Applies in sentence mode only.
            append_trailing_space=True,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._receive_task = None

        # Per-connection state. Only the last utterance can still take text.
        self._utterances: list[_Utterance] = []
        # Contexts the base class will send no more text for; each closes with its last utterance.
        self._ended_context_ids: set[str] = set()
        # Until a cancel is acknowledged, everything received belongs to cancelled work, so a
        # new utterance waits for the acknowledgement.
        self._unacked_cancels = 0
        self._cancels_acked = asyncio.Event()
        self._utterance_started = asyncio.Event()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Kalpa service supports metrics generation.
        """
        return True

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and open the Kalpa session.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._kalpa_sample_rate = _resolve_sample_rate(self.sample_rate)
        await self._connect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket:
            self._create_keepalive_task(interval=_KEEPALIVE_INTERVAL_SECONDS)

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._cancel_keepalive_task()

        await self._disconnect_websocket()

    async def _reopen(self):
        """Replace the connection, keeping the audio context of the reply about to be sent."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"{self} failed to close Kalpa websocket: {e}")
            self._websocket = None
        await self._connect()

    async def _connect_websocket(self):
        """Open the socket and hold the session at ``sessionCreated``."""
        websocket = None
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Kalpa")

            websocket = await self._websocket_connect(
                f"{self._base_url}/v1/tts/{self._settings.voice}/stream"
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "initializeConnection",
                        "api_key": self._api_key,
                        "model": self._settings.model,
                        # Start a reply's audio at its first complete sentences.
                        "generation_config": {},
                        "output_format": "pcm_s16le",
                        "sample_rate": self._kalpa_sample_rate,
                    }
                )
            )

            # The key, voice and model are checked before `sessionCreated`, so a
            # rejected session fails here rather than on the first reply.
            message = json.loads(
                await asyncio.wait_for(websocket.recv(), timeout=_SESSION_TIMEOUT_SECONDS)
            )
            if message.get("type") == "error":
                await self._push_kalpa_error(message)
                await websocket.close()
                self._websocket = None
                await self._call_event_handler("on_connection_error", message["error"]["message"])
                return
            if message.get("type") != "sessionCreated":
                raise Exception(f"unexpected message before sessionCreated: {message}")

            logger.debug(f"{self}: session created (session_id: {message.get('session_id')})")
            # The rate Kalpa renders at; the output transport resamples any difference.
            self._kalpa_sample_rate = message["sample_rate"]
            self._utterances.clear()
            self._unacked_cancels = 0
            self._cancels_acked.set()
            self._utterance_started.set()
            self._websocket = websocket
            await self._call_event_handler("on_connected")
        except BaseException as e:
            if websocket is not None:
                try:
                    await websocket.close()
                except Exception:
                    pass
            if not isinstance(e, Exception):
                raise
            await self.push_error(error_msg=f"{self} error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Kalpa")
                await self._websocket.close()
        except Exception as e:
            logger.debug(f"{self} failed to close Kalpa websocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Args:
            delta: A :class:`TTSSettings` (or ``KalpaTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        # The voice and model are fixed for the life of a connection.
        if changed.keys() & {"voice", "model"}:
            await self._disconnect()
            await self._connect()

        return changed

    async def _send_keepalive(self):
        """Reset Kalpa's idle timeout; empty text without a flush does nothing else."""
        await self._get_websocket().send(json.dumps({"type": "sendText", "text": ""}))

    def _open_utterance(self) -> _Utterance | None:
        if self._utterances and not self._utterances[-1].flushed:
            return self._utterances[-1]
        return None

    async def _flush(self, utterance: _Utterance):
        """End the utterance, so Kalpa speaks the rest of its text."""
        # Kalpa holds one flushed utterance until it can start it, and rejects a flush while
        # one waits. Its `responseCreated` shows it has started.
        while any(u.flushed and not u.response_id for u in self._utterances):
            self._utterance_started.clear()
            try:
                await asyncio.wait_for(self._utterance_started.wait(), _FLUSH_WAIT_SECONDS)
            except TimeoutError:
                break
        if utterance not in self._utterances:
            return  # cancelled while waiting
        utterance.flushed = True
        await self._get_websocket().send(json.dumps({"type": "sendText", "flush": True}))

    async def _close_if_finished(self, context_id: str):
        """Close a context once the base class has ended it and its last utterance is done."""
        if context_id not in self._ended_context_ids:
            return
        if any(u.context_id == context_id for u in self._utterances):
            return
        self._ended_context_ids.discard(context_id)
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def _cancel_replies(self) -> list[str]:
        """Drop everything Kalpa holds for this connection: buffered text, queued and generating replies.

        Returns:
            The contexts that were cancelled.
        """
        context_ids = [u.context_id for u in self._utterances]
        if not context_ids:
            return []
        self._utterances.clear()
        self._ended_context_ids.difference_update(context_ids)
        self._utterance_started.set()
        self._unacked_cancels += 1
        self._cancels_acked.clear()
        await self._get_websocket().send(json.dumps({"type": "cancelResponse"}))
        return context_ids

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the interrupted reply on the server instead of reconnecting."""
        await self.stop_all_metrics()
        try:
            await self._cancel_replies()
        except Exception as e:
            logger.error(f"{self} error sending cancelResponse: {e}")
        await super().on_audio_context_interrupted(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """End the reply so Kalpa speaks the rest of its text.

        Args:
            context_id: The reply to end. Falls back to the open one.
        """
        utterance = self._open_utterance()
        context_id = context_id or (utterance.context_id if utterance else None)
        if not context_id:
            return
        self._ended_context_ids.add(context_id)
        try:
            if utterance and utterance.context_id == context_id:
                await self._flush(utterance)
        except Exception as e:
            logger.error(f"{self} error sending flush: {e}")
        await self._close_if_finished(context_id)

    async def _push_kalpa_error(self, message: dict):
        error = message.get("error", {})
        category = _error_category(error.get("type"))
        await self.push_error(
            error_msg=f"{self} error ({error.get('type')}): {error.get('message')}",
            category=category,
            # The connection is closing. Only a server failure is worth reconnecting for.
            force_treat_as_permanent=bool(message.get("fatal"))
            and category not in (ErrorCategory.SERVER, ErrorCategory.RATE_LIMIT),
        )

    async def _handle_error(self, message: dict):
        if message.get("response_id") and not message.get("fatal") and self._unacked_cancels:
            return  # the failure of a cancelled reply

        await self._push_kalpa_error(message)

        if not message.get("fatal") and not message.get("response_id"):
            # Kalpa rejected a message and left everything else as it was, so its utterances
            # may no longer line up with ours. Drop them all and start over.
            for context_id in await self._cancel_replies():
                if self.audio_context_available(context_id):
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)

    async def _receive_messages(self):
        async for raw in self._get_websocket():
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON message: {raw}")
                continue

            msg_type = message.get("type")
            response_id = message.get("response_id")
            utterance = next((u for u in self._utterances if u.response_id == response_id), None)

            if msg_type == "cancelled":
                self._unacked_cancels = max(self._unacked_cancels - 1, 0)
                if not self._unacked_cancels:
                    self._cancels_acked.set()
            elif msg_type == "error":
                await self._handle_error(message)
            elif msg_type == "responseCreated":
                unnamed = next((u for u in self._utterances if u.response_id is None), None)
                if unnamed:
                    unnamed.response_id = response_id
                    self._utterance_started.set()
            elif msg_type == "responseAudio" and utterance:
                audio = base64.b64decode(message["pcm_b64"])
                await self.append_to_audio_context(
                    utterance.context_id,
                    TTSAudioRawFrame(
                        audio, self._kalpa_sample_rate, 1, context_id=utterance.context_id
                    ),
                )
            elif msg_type == "responseDone" and utterance:
                # Also ends an utterance that failed before its flush; more text for its
                # context then starts a new one.
                self._utterances.remove(utterance)
                await self._close_if_finished(utterance.context_id)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Append text to the reply's utterance.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Nothing directly; audio arrives on the receive task.
        """
        try:
            if self._unacked_cancels:
                try:
                    await asyncio.wait_for(self._cancels_acked.wait(), _CANCEL_ACK_TIMEOUT_SECONDS)
                except TimeoutError:
                    logger.warning(f"{self}: Kalpa did not acknowledge a cancel; reconnecting")
            if (
                self._unacked_cancels
                or not self._websocket
                or self._websocket.state is State.CLOSED
            ):
                await self._reopen()

            utterance = self._open_utterance()
            if utterance and (
                utterance.context_id != context_id
                or utterance.chars + len(text) > _MAX_UTTERANCE_CHARS
            ):
                # Kalpa buffers one utterance at a time, so another context's text (a
                # TTSSpeakFrame mid-reply), or text past an utterance's limit, ends it first.
                await self._flush(utterance)
                utterance = None
            if not utterance:
                utterance = _Utterance(context_id)
                self._utterances.append(utterance)
            utterance.chars += len(text)

            await self._get_websocket().send(json.dumps({"type": "sendText", "text": text}))
            await self.start_tts_usage_metrics(text)

            # The audio frames will be handled in _receive_messages
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"{self} error: {e}", exception=e)


class KalpaHttpTTSService(TTSService):
    """Kalpa HTTP text-to-speech service.

    Sends each sentence to ``POST /v1/tts/{voice_id}/stream``, which returns raw
    PCM as it is generated, and plays the body as it arrives. Voice agents
    should prefer :class:`KalpaTTSService`, which streams a whole reply over
    one connection.
    """

    Settings = KalpaTTSSettings
    _settings: Settings
    # The rate Kalpa synthesizes at, resolved in setup() from the pipeline's.
    _kalpa_sample_rate: int

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.kalpalabs.ai",
        sample_rate: int | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Kalpa HTTP TTS service.

        Args:
            api_key: Kalpa API key for authentication.
            base_url: Base URL for the Kalpa API. Defaults to
                ``https://api.kalpalabs.ai``.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline
                default.
            aiohttp_session: Optional shared aiohttp session. When omitted, the
                service creates and owns one.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to ``TTSService``.
        """
        if not api_key:
            raise ValueError("Kalpa API key is required")
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Kalpa service supports metrics generation.
        """
        return True

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service, creating an aiohttp session if one was not provided.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._kalpa_sample_rate = _resolve_sample_rate(self.sample_rate)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame: EndFrame):
        """Stop the service and release an owned aiohttp session.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and release an owned aiohttp session.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._close_session()

    async def cleanup(self):
        """Release Kalpa TTS resources at teardown."""
        await super().cleanup()
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using ``POST /v1/tts/{voice_id}/stream``.

        Args:
            text: The text to synthesize.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        payload = {
            "text": text,
            "model": self._settings.model,
            "output_format": "pcm_s16le",
            "sample_rate": self._kalpa_sample_rate,
        }

        try:
            async with self._session.post(
                f"{self._base_url}/v1/tts/{self._settings.voice}/stream",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            ) as response:
                if response.status != 200:
                    await self._push_http_error(response)
                    return

                await self.start_tts_usage_metrics(text)

                # A body cut off after audio started raises, so it isn't taken for a whole clip.
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=self._kalpa_sample_rate,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"{self} error: {e}", exception=e)

    async def _push_http_error(self, response: aiohttp.ClientResponse):
        """Report a non-200 response from its ``{"error": {"type", "message"}}`` envelope."""
        try:
            error = (await response.json(content_type=None))["error"]
            if not isinstance(error, dict):
                raise TypeError(error)
        except Exception:
            error = {"message": (await response.text(errors="ignore"))[:200]}
        category = _error_category(error.get("type"), response.status)
        await self.push_error(
            error_msg=f"{self} error (status {response.status}, {error.get('type')}): "
            f"{error.get('message')}",
            category=category,
            # An exhausted allowance doesn't come back with time.
            force_treat_as_permanent=category == ErrorCategory.QUOTA,
        )
