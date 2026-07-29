"""Tests for the Wyoming event handling."""
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info

from wyoming_parakeet.engine import SAMPLE_RATE
from wyoming_parakeet.handler import (
    DEFAULT_MAX_AUDIO_SECONDS,
    ParakeetEventHandler,
)

ARGS = Namespace(max_audio_seconds=DEFAULT_MAX_AUDIO_SECONDS)

INFO = Info(
    asr=[
        AsrProgram(
            name="parakeet-mlx",
            description="test",
            attribution=Attribution(name="t", url="http://example.invalid"),
            installed=True,
            version="1.0.0",
            models=[
                AsrModel(
                    name="fake/model",
                    description="test",
                    attribution=Attribution(name="t", url="http://example.invalid"),
                    installed=True,
                    version=None,
                    languages=["en"],
                )
            ],
        )
    ]
)


@pytest.fixture
def handler():
    engine = MagicMock()
    engine.transcribe = AsyncMock(return_value="turn off the kitchen lights")
    h = ParakeetEventHandler(INFO, ARGS, engine, MagicMock(), MagicMock())
    h.write_event = AsyncMock()
    return h


def chunk(n_samples, rate=SAMPLE_RATE):
    return AudioChunk(
        rate=rate, width=2, channels=1, audio=b"\x01\x00" * n_samples
    ).event()


async def test_describe_returns_info(handler):
    assert await handler.handle_event(Describe().event()) is True
    handler.write_event.assert_awaited_once()
    assert Info.is_type(handler.write_event.await_args[0][0].type)


async def test_transcribe_event_is_accepted(handler):
    assert await handler.handle_event(Transcribe(language="en").event()) is True
    handler.write_event.assert_not_awaited()


async def test_unknown_event_keeps_connection_open(handler):
    from wyoming.event import Event

    assert await handler.handle_event(Event(type="something-else")) is True


async def test_chunks_accumulate_before_stop(handler):
    for _ in range(3):
        assert await handler.handle_event(chunk(100)) is True
    handler.engine.transcribe.assert_not_awaited()
    assert len(handler.audio) == 3 * 100 * 2


async def test_stop_transcribes_accumulated_audio(handler):
    await handler.handle_event(chunk(SAMPLE_RATE // 2))
    await handler.handle_event(chunk(SAMPLE_RATE // 2))

    assert await handler.handle_event(AudioStop().event()) is False

    handler.engine.transcribe.assert_awaited_once()
    assert len(handler.engine.transcribe.await_args[0][0]) == SAMPLE_RATE * 2


async def test_stop_writes_transcript(handler):
    await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())

    event = handler.write_event.await_args[0][0]
    assert Transcript.is_type(event.type)
    assert Transcript.from_event(event).text == "turn off the kitchen lights"


async def test_empty_transcript_is_still_sent(handler):
    """Silence must produce an empty Transcript, not a dropped response --
    Home Assistant waits for one."""
    handler.engine.transcribe = AsyncMock(return_value="")
    await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())

    event = handler.write_event.await_args[0][0]
    assert Transcript.is_type(event.type)
    assert Transcript.from_event(event).text == ""


async def test_buffer_resets_after_stop(handler):
    await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())
    assert handler.audio == b""


async def test_stop_with_no_audio_does_not_crash(handler):
    assert await handler.handle_event(AudioStop().event()) is False
    handler.engine.transcribe.assert_awaited_once_with(b"")


async def test_resampled_input_is_converted_to_16k(handler):
    """Satellites may send other rates; the converter must normalise them
    before the engine sees them."""
    await handler.handle_event(chunk(48000, rate=48000))
    await handler.handle_event(AudioStop().event())

    pcm = handler.engine.transcribe.await_args[0][0]
    assert len(pcm) == SAMPLE_RATE * 2, "1s of 48kHz audio should become 1s at 16kHz"


# --- failure handling -------------------------------------------------------
# wyoming's run loop is try/finally with no except, so an exception escaping
# handle_event closes the connection having sent nothing and Home Assistant
# waits for a response that never arrives.


async def test_model_failure_still_sends_a_transcript(handler):
    handler.engine.transcribe = AsyncMock(side_effect=RuntimeError("metal exploded"))

    assert await handler.handle_event(AudioStop().event()) is False

    event = handler.write_event.await_args[0][0]
    assert Transcript.is_type(event.type)
    assert Transcript.from_event(event).text == ""


async def test_model_failure_does_not_propagate(handler):
    """Must not escape into wyoming's run loop."""
    handler.engine.transcribe = AsyncMock(side_effect=RuntimeError("metal exploded"))
    await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())  # would raise if unhandled


async def test_model_failure_is_logged_with_traceback(handler, caplog):
    handler.engine.transcribe = AsyncMock(side_effect=RuntimeError("metal exploded"))
    with caplog.at_level("ERROR"):
        await handler.handle_event(AudioStop().event())
    assert "metal exploded" in caplog.text


async def test_buffer_resets_after_failure(handler):
    handler.engine.transcribe = AsyncMock(side_effect=RuntimeError("metal exploded"))
    await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())
    assert handler.audio == b""


# --- isolation --------------------------------------------------------------
# The whisper.cpp server this replaced leaked decoder context between requests
# and would return the *previous* utterance. Guard against reintroducing any
# shared per-request state.


async def test_concurrent_handlers_do_not_share_audio():
    engine = MagicMock()
    engine.transcribe = AsyncMock(return_value="")
    seen = []
    engine.transcribe.side_effect = lambda pcm: seen.append(pcm) or ""

    a = ParakeetEventHandler(INFO, ARGS, engine, MagicMock(), MagicMock())
    b = ParakeetEventHandler(INFO, ARGS, engine, MagicMock(), MagicMock())
    a.write_event = AsyncMock()
    b.write_event = AsyncMock()

    # Interleave two conversations through one shared engine.
    await a.handle_event(chunk(100))
    await b.handle_event(chunk(300))
    await a.handle_event(chunk(100))
    await a.handle_event(AudioStop().event())
    await b.handle_event(AudioStop().event())

    assert [len(p) for p in seen] == [200 * 2, 300 * 2]


async def test_transcript_reflects_only_this_requests_audio(handler):
    """A second utterance must not inherit the first one's text."""
    handler.engine.transcribe = AsyncMock(side_effect=["first", "second"])

    await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())
    first = Transcript.from_event(handler.write_event.await_args[0][0]).text

    await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())
    second = Transcript.from_event(handler.write_event.await_args[0][0]).text

    assert (first, second) == ("first", "second")


# --- audio format normalisation ---------------------------------------------
# Satellites vary; the engine must always receive 16 kHz mono 16-bit.


async def test_stereo_input_is_downmixed(handler):
    stereo = AudioChunk(
        rate=SAMPLE_RATE, width=2, channels=2, audio=b"\x01\x00\x01\x00" * SAMPLE_RATE
    ).event()
    await handler.handle_event(stereo)
    await handler.handle_event(AudioStop().event())

    assert len(handler.engine.transcribe.await_args[0][0]) == SAMPLE_RATE * 2


async def test_8bit_input_is_widened(handler):
    narrow = AudioChunk(
        rate=SAMPLE_RATE, width=1, channels=1, audio=b"\x40" * SAMPLE_RATE
    ).event()
    await handler.handle_event(narrow)
    await handler.handle_event(AudioStop().event())

    assert len(handler.engine.transcribe.await_args[0][0]) == SAMPLE_RATE * 2


# --- resource limits --------------------------------------------------------
# Wyoming is unauthenticated, so buffered audio is attacker-controlled.


def capped_handler(seconds):
    engine = MagicMock()
    engine.transcribe = AsyncMock(return_value="ok")
    h = ParakeetEventHandler(
        INFO, Namespace(max_audio_seconds=seconds), engine, MagicMock(), MagicMock()
    )
    h.write_event = AsyncMock()
    return h


async def test_audio_buffer_is_capped():
    handler = capped_handler(1.0)
    for _ in range(10):
        await handler.handle_event(chunk(SAMPLE_RATE))  # 10s into a 1s cap
    assert len(handler.audio) == SAMPLE_RATE * 2


async def test_truncated_audio_is_still_transcribed():
    """Cap the memory, don't drop the user's command on the floor."""
    handler = capped_handler(1.0)
    for _ in range(5):
        await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())

    handler.engine.transcribe.assert_awaited_once()
    assert len(handler.engine.transcribe.await_args[0][0]) == SAMPLE_RATE * 2
    assert Transcript.from_event(handler.write_event.await_args[0][0]).text == "ok"


async def test_truncation_warns_once(caplog):
    handler = capped_handler(1.0)
    with caplog.at_level("WARNING"):
        for _ in range(6):
            await handler.handle_event(chunk(SAMPLE_RATE))
    assert caplog.text.count("Audio exceeded") == 1


async def test_cap_resets_between_utterances():
    handler = capped_handler(1.0)
    for _ in range(3):
        await handler.handle_event(chunk(SAMPLE_RATE))
    await handler.handle_event(AudioStop().event())
    assert handler.truncated is False

    await handler.handle_event(chunk(SAMPLE_RATE // 2))
    assert len(handler.audio) == SAMPLE_RATE  # accepted again, not still capped


async def test_audio_under_the_cap_is_untouched():
    handler = capped_handler(DEFAULT_MAX_AUDIO_SECONDS)
    await handler.handle_event(chunk(SAMPLE_RATE * 3))
    assert len(handler.audio) == SAMPLE_RATE * 3 * 2


# --- transcript privacy -----------------------------------------------------
# Logs are long-lived and readable by other local accounts.


async def test_transcript_text_is_not_logged_at_info(handler, caplog):
    handler.engine.transcribe = AsyncMock(return_value="unlock the front door")
    with caplog.at_level("INFO"):
        await handler.handle_event(chunk(SAMPLE_RATE))
        await handler.handle_event(AudioStop().event())
    assert "unlock the front door" not in caplog.text


async def test_transcript_text_is_available_at_debug(handler, caplog):
    handler.engine.transcribe = AsyncMock(return_value="unlock the front door")
    with caplog.at_level("DEBUG"):
        await handler.handle_event(chunk(SAMPLE_RATE))
        await handler.handle_event(AudioStop().event())
    assert "unlock the front door" in caplog.text
