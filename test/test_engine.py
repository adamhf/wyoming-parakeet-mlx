"""Tests for ParakeetEngine.

The model itself is mocked throughout -- these cover the audio marshalling
and threading around it, which is where the real bugs were.
"""
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wyoming_parakeet.engine import MIN_SAMPLES, SAMPLE_RATE, ParakeetEngine


def make_engine(text="turn off the kitchen lights"):
    engine = ParakeetEngine("fake/model")
    result = MagicMock()
    result.text = text
    engine.model = MagicMock()
    engine.model.generate.return_value = [result]
    return engine


def test_returns_model_text(pcm):
    engine = make_engine("set a timer for 12 minutes")
    with patch("wyoming_parakeet.engine.get_logmel"):
        assert engine._transcribe(pcm(SAMPLE_RATE)) == "set a timer for 12 minutes"


def test_short_audio_short_circuits_without_touching_model(pcm):
    """A clipped VAD flush must not reach the model at all."""
    engine = make_engine()
    with patch("wyoming_parakeet.engine.get_logmel") as get_logmel:
        assert engine._transcribe(pcm(MIN_SAMPLES - 1)) == ""
    engine.model.generate.assert_not_called()
    get_logmel.assert_not_called()


def test_audio_at_threshold_is_processed(pcm):
    engine = make_engine()
    with patch("wyoming_parakeet.engine.get_logmel"):
        assert engine._transcribe(pcm(MIN_SAMPLES)) != ""
    engine.model.generate.assert_called_once()


def test_empty_audio_returns_empty_string():
    engine = make_engine()
    with patch("wyoming_parakeet.engine.get_logmel"):
        assert engine._transcribe(b"") == ""


def test_no_results_returns_empty_string(pcm):
    """Silence legitimately decodes to nothing; don't IndexError on it."""
    engine = make_engine()
    engine.model.generate.return_value = []
    with patch("wyoming_parakeet.engine.get_logmel"):
        assert engine._transcribe(pcm(SAMPLE_RATE)) == ""


def test_audio_is_float32_not_bfloat16(pcm):
    """Regression: get_logmel views the complex STFT output as the input
    dtype, so anything narrower than float32 silently doubles the mel bin
    count and the downstream matmul fails."""
    engine = make_engine()
    with patch("wyoming_parakeet.engine.get_logmel") as get_logmel:
        engine._transcribe(pcm(SAMPLE_RATE))
    samples = get_logmel.call_args[0][0]
    assert np.asarray(samples).dtype == np.float32


def test_pcm_is_scaled_to_unit_range(pcm):
    engine = make_engine()
    with patch("wyoming_parakeet.engine.get_logmel") as get_logmel:
        engine._transcribe(pcm(SAMPLE_RATE, value=16384))
    samples = np.asarray(get_logmel.call_args[0][0])
    assert samples.shape == (SAMPLE_RATE,)
    assert np.allclose(samples, 0.5)


def test_full_scale_pcm_stays_within_unit_range(pcm):
    engine = make_engine()
    with patch("wyoming_parakeet.engine.get_logmel") as get_logmel:
        engine._transcribe(pcm(SAMPLE_RATE, value=-32768))
    samples = np.asarray(get_logmel.call_args[0][0])
    assert np.abs(samples).max() <= 1.0


def test_preprocessor_config_is_passed_through(pcm):
    engine = make_engine()
    with patch("wyoming_parakeet.engine.get_logmel") as get_logmel:
        engine._transcribe(pcm(SAMPLE_RATE))
    assert get_logmel.call_args[0][1] is engine.model.preprocessor_config


@pytest.mark.asyncio
async def test_load_and_inference_share_one_thread():
    """Regression: MLX streams are thread-local, so a model loaded on one
    thread cannot be evaluated from another -- mx.eval() raises
    'There is no Stream(cpu, 1) in current thread'."""
    import asyncio

    engine = ParakeetEngine("fake/model")
    threads = []
    lock = threading.Lock()

    def record():
        with lock:
            threads.append(threading.get_ident())
        # Hold the worker. Sequential calls can coincidentally reuse a single
        # thread out of a multi-worker pool, so overlap them -- a pool wider
        # than one will hand these to different threads and fail the assert.
        threading.Event().wait(0.05)

    engine._load = record
    engine._transcribe = lambda _pcm: (record(), "")[1]

    await engine.start()
    await asyncio.gather(*(engine.transcribe(b"") for _ in range(4)))

    assert len(threads) == 5
    assert len(set(threads)) == 1, "load and inference must share one thread"
    assert threads[0] != threading.get_ident(), "must not run on the event loop"


@pytest.mark.asyncio
async def test_requests_are_serialised(pcm):
    """A single worker means overlapping satellites queue rather than
    racing the ANE."""
    import asyncio

    engine = ParakeetEngine("fake/model")
    concurrent = 0
    peak = 0

    def slow(_pcm):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        threading.Event().wait(0.05)
        concurrent -= 1
        return "ok"

    engine._transcribe = slow
    await asyncio.gather(*(engine.transcribe(pcm(SAMPLE_RATE)) for _ in range(4)))
    assert peak == 1


def test_transcription_carries_no_state_between_calls(pcm):
    """Regression guard for the bug class that motivated leaving whisper.cpp:
    its server reused decoder context across requests and would return the
    previous utterance. Each call here must stand alone."""
    engine = make_engine()
    audio = pcm(SAMPLE_RATE)

    with patch("wyoming_parakeet.engine.get_logmel") as get_logmel:
        engine._transcribe(pcm(SAMPLE_RATE * 2, value=4096))
        engine._transcribe(audio)
        first = np.asarray(get_logmel.call_args[0][0])

        engine._transcribe(pcm(SAMPLE_RATE // 2, value=-2048))
        engine._transcribe(audio)
        second = np.asarray(get_logmel.call_args[0][0])

    assert np.array_equal(first, second)
    # generate() must be handed only the current mel, with no prompt/context.
    assert engine.model.generate.call_args[0][0] is get_logmel.return_value
    assert engine.model.generate.call_args.kwargs == {}


@pytest.mark.asyncio
async def test_start_warms_the_model():
    """The first inference JITs Metal kernels. If warm-up is dropped, the
    first voice command after every reboot pays that cost."""
    engine = ParakeetEngine("fake/model")
    model = MagicMock()
    model.generate.return_value = []

    with patch("parakeet_mlx.from_pretrained", return_value=model) as load:
        with patch("wyoming_parakeet.engine.get_logmel"):
            await engine.start()

    load.assert_called_once_with("fake/model")
    assert model.generate.call_count == 1, "start() should run one warm-up pass"


@pytest.mark.asyncio
async def test_engine_survives_a_failed_request(pcm):
    """One bad utterance must not poison the worker for later ones."""
    engine = make_engine("recovered")
    with patch("wyoming_parakeet.engine.get_logmel", side_effect=[RuntimeError("boom"), MagicMock()]):
        with pytest.raises(RuntimeError):
            await engine.transcribe(pcm(SAMPLE_RATE))
        assert await engine.transcribe(pcm(SAMPLE_RATE)) == "recovered"
