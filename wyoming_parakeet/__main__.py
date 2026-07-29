#!/usr/bin/env python3
"""Wyoming ASR server for NVIDIA Parakeet via parakeet-mlx (Apple Silicon)."""
import argparse
import asyncio
import logging

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from .engine import ParakeetEngine
from .handler import DEFAULT_MAX_AUDIO_SECONDS, ParakeetEventHandler

_LOGGER = logging.getLogger(__name__)
__version__ = "1.0.0"

# v2 is English-only but does better inverse text normalisation than the
# multilingual v3 ("21 degrees" / "30%" rather than "twenty-one degrees" /
# "thirty percent"), which is what Home Assistant's local intent matching
# expects. Don't switch to v3 without re-checking that.
DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id")
    parser.add_argument("--language", default="en", help="Language code reported to HA")
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=DEFAULT_MAX_AUDIO_SECONDS,
        help="Cap on buffered audio per utterance (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log DEBUG messages, including transcript text",
    )
    parser.add_argument("--log-format", default=logging.BASIC_FORMAT)
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def build_info(model: str, language: str) -> Info:
    """Describe this service to Home Assistant's Wyoming config flow."""
    return Info(
        asr=[
            AsrProgram(
                name="parakeet-mlx",
                description="NVIDIA Parakeet TDT via MLX",
                attribution=Attribution(
                    name="senstella", url="https://github.com/senstella/parakeet-mlx"
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model,
                        description=model,
                        attribution=Attribution(
                            name="NVIDIA",
                            url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
                        ),
                        installed=True,
                        version=None,
                        languages=[language],
                    )
                ],
            )
        ]
    )


async def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )

    _LOGGER.info("Loading %s", args.model)
    engine = ParakeetEngine(args.model)
    await engine.start()

    wyoming_info = build_info(args.model, args.language)
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready on %s", args.uri)
    await server.run(
        lambda *a, **kw: ParakeetEventHandler(wyoming_info, args, engine, *a, **kw)
    )


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
