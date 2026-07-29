"""Tests for CLI wiring and the Info advertised to Home Assistant."""
from wyoming.info import Info

from wyoming_parakeet.__main__ import DEFAULT_MODEL, build_info, build_parser


def test_default_model_is_v2_not_v3():
    """Deliberate: v3 is newer and multilingual but spells numbers out
    ("twenty-one degrees" vs "21 degrees"). Home Assistant's local intent
    matching wants digits, and both pipelines run prefer_local_intents, so v3
    would still look accurate while pushing commands onto the LLM fallback.
    If you are changing this, re-run test/wy-test.py and check cmd2/cmd4/cmd7."""
    assert DEFAULT_MODEL == "mlx-community/parakeet-tdt-0.6b-v2"


def test_model_defaults_are_applied():
    args = build_parser().parse_args(["--uri", "tcp://0.0.0.0:7892"])
    assert args.model == DEFAULT_MODEL
    assert args.language == "en"
    assert args.debug is False


def test_model_can_be_overridden():
    args = build_parser().parse_args(
        ["--uri", "tcp://0.0.0.0:7892", "--model", "mlx-community/other"]
    )
    assert args.model == "mlx-community/other"


def test_info_advertises_the_running_model():
    """Home Assistant's Wyoming config flow reads this; if it is malformed the
    integration cannot be added at all."""
    info = build_info("mlx-community/other", "en")
    assert Info.is_type(info.event().type)

    program = info.asr[0]
    assert program.installed is True
    assert program.models[0].name == "mlx-community/other"
    assert program.models[0].languages == ["en"]


def test_info_round_trips_through_an_event():
    info = build_info(DEFAULT_MODEL, "en")
    restored = Info.from_event(info.event())
    assert restored.asr[0].models[0].name == DEFAULT_MODEL
