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


class ParakeetEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, cli_args, engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.engine = engine
        self.audio = bytes()
        self.converter = AudioChunkConverter(rate=SAMPLE_RATE, width=2, channels=1)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            if not self.audio:
                _LOGGER.debug("Receiving audio")
            self.audio += self.converter.convert(AudioChunk.from_event(event)).audio
            return True

        if AudioStop.is_type(event.type):
            duration = len(self.audio) / (SAMPLE_RATE * 2)
            started = time.monotonic()
            try:
                text = await self.engine.transcribe(self.audio)
                _LOGGER.info(
                    "%.2fs audio -> %.0fms :: %r",
                    duration,
                    (time.monotonic() - started) * 1000,
                    text,
                )
            except Exception:
                # wyoming's run loop has no except clause, so letting this
                # propagate closes the connection without ever sending a
                # Transcript -- Home Assistant then waits for a response that
                # will never arrive. Fail fast with an empty result instead.
                _LOGGER.exception("Transcription failed after %.2fs audio", duration)
                text = ""
            finally:
                self.audio = bytes()

            await self.write_event(Transcript(text=text).event())
            return False

        if Transcribe.is_type(event.type):
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
