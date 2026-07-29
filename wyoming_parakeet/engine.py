"""Parakeet model wrapper pinned to a single MLX worker thread."""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
from parakeet_mlx.audio import get_logmel

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Shorter than this and the encoder has nothing useful to chew on; Wyoming
# clients occasionally flush a near-empty buffer when VAD clips too tightly.
MIN_SAMPLES = SAMPLE_RATE // 10


class ParakeetEngine:
    """Owns the model and guarantees every MLX call happens on one thread.

    MLX streams are thread-local, so a model loaded on the main thread cannot
    be evaluated from an arbitrary executor thread -- mx.eval() raises
    "There is no Stream(cpu, 1) in current thread". Loading and inference both
    run on this single worker, which also serialises requests for free.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    def _run(self, fn, *args):
        return asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _load(self) -> None:
        from parakeet_mlx import from_pretrained

        started = time.monotonic()
        self.model = from_pretrained(self.model_name)
        _LOGGER.info("Loaded %s in %.1fs", self.model_name, time.monotonic() - started)
        # First inference JITs Metal kernels; pay that now, not on the user's
        # first voice command.
        started = time.monotonic()
        self._transcribe(b"\x00\x00" * SAMPLE_RATE)
        _LOGGER.info("Warmed up in %.1fs", time.monotonic() - started)

    def _transcribe(self, pcm: bytes) -> str:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size < MIN_SAMPLES:
            return ""
        # parakeet_mlx.load_audio() shells out to ffmpeg, which we neither have
        # nor need: Wyoming already hands us 16 kHz mono PCM, so build the mel
        # directly. float32 is required -- get_logmel views the complex STFT
        # output as the input dtype, so bfloat16 silently doubles the bin count.
        mel = get_logmel(mx.array(samples), self.model.preprocessor_config)
        results = self.model.generate(mel)
        return results[0].text if results else ""

    async def start(self) -> None:
        await self._run(self._load)

    async def transcribe(self, pcm: bytes) -> str:
        return await self._run(self._transcribe, pcm)
