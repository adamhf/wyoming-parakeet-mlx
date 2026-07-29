"""Wyoming event handler backed by an in-process parakeet-mlx model."""
import logging
import time

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .engine import SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)

# Wyoming has no authentication, so any client that can reach the port can
# stream audio. Without a cap, self.audio grows until AudioStop -- measured at
# ~11 MB/s over loopback, which exhausts 32 GB in under an hour from a single
# connection. A stuck satellite that never sends AudioStop does the same thing
# by accident. No real voice command approaches this bound.
DEFAULT_MAX_AUDIO_SECONDS = 120


class ParakeetEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, cli_args, engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.engine = engine
        self.audio = bytes()
        self.truncated = False
        self.converter = AudioChunkConverter(rate=SAMPLE_RATE, width=2, channels=1)
        max_seconds = getattr(cli_args, "max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS)
        self.max_bytes = int(max_seconds * SAMPLE_RATE * 2)

    def _append(self, chunk: bytes) -> None:
        room = self.max_bytes - len(self.audio)
        if room > 0:
            self.audio += chunk[:room]
        if len(self.audio) >= self.max_bytes and not self.truncated:
            self.truncated = True
            _LOGGER.warning(
                "Audio exceeded %.0fs; ignoring the rest of this utterance",
                self.max_bytes / (SAMPLE_RATE * 2),
            )

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            if not self.audio:
                _LOGGER.debug("Receiving audio")
            self._append(self.converter.convert(AudioChunk.from_event(event)).audio)
            return True

        if AudioStop.is_type(event.type):
            duration = len(self.audio) / (SAMPLE_RATE * 2)
            started = time.monotonic()
            try:
                text = await self.engine.transcribe(self.audio)
                # The transcript is everything the user said, and the log file
                # is long-lived and readable by other local accounts. Keep the
                # operational signal at INFO and the content behind --debug.
                _LOGGER.info(
                    "%.2fs audio -> %.0fms, %d chars",
                    duration,
                    (time.monotonic() - started) * 1000,
                    len(text),
                )
                _LOGGER.debug("Transcript: %r", text)
            except Exception:
                # wyoming's run loop has no except clause, so letting this
                # propagate closes the connection without ever sending a
                # Transcript -- Home Assistant then waits for a response that
                # will never arrive. Fail fast with an empty result instead.
                _LOGGER.exception("Transcription failed after %.2fs audio", duration)
                text = ""
            finally:
                self.audio = bytes()
                self.truncated = False

            await self.write_event(Transcript(text=text).event())
            return False

        if Transcribe.is_type(event.type):
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
