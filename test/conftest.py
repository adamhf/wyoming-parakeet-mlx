import numpy as np
import pytest


@pytest.fixture
def pcm():
    """Build int16 mono PCM bytes of a given sample count."""

    def _pcm(n_samples: int, value: int = 8192) -> bytes:
        return np.full(n_samples, value, dtype=np.int16).tobytes()

    return _pcm
