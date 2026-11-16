import os
import re

from dataclasses import dataclass
from dotenv import load_dotenv


# Load variables from the .env file.
load_dotenv()


@dataclass(frozen=True)
class Settings:

    # BITalino connection
    BITALINO_MAC_ADDRESS: str = os.getenv(
        "BITALINO_MAC_ADDRESS", ""
    )

    # Acquisition settings
    SAMPLING_RATE: int = int(
        os.getenv("SAMPLING_RATE", "1000")
    )

    ANALOG_CHANNELS: tuple = tuple(
        int(channel)
        for channel in os.getenv(
            "ANALOG_CHANNELS", "3"
        ).split(",")
    )

    BATCH_SAMPLES: int = int(
        os.getenv(
            "BATCH_SAMPLES",
            str(max(1, round(SAMPLING_RATE * 0.05)))
        )
    )

    # Signal-processing settings
    WINDOW_SECONDS: float = float(
        os.getenv("WINDOW_SECONDS", "1.0")
    )

    INFERENCE_STRIDE_MS: int = int(
        os.getenv("INFERENCE_STRIDE_MS", "50")
    )

    # Model settings
    BLINK_THRESHOLD: float = float(
        os.getenv("BLINK_THRESHOLD", "0.5")
    )

    MODEL_MODULE: str = os.getenv(
        "MODEL_MODULE", "blink_model"
    )

    MODEL_FUNCTION: str = os.getenv(
        "MODEL_FUNCTION", "predict_blink"
    )


settings = Settings()


def validate_settings():

    # Check the Bluetooth MAC address.
    if not re.fullmatch(
        r"[0-9A-Fa-f]{2}([:-][0-9A-Fa-f]{2}){5}",
        settings.BITALINO_MAC_ADDRESS
    ):
        raise ValueError(
            "Invalid BITalino Bluetooth MAC address."
        )

    # BITalino-supported sampling rates.
    if settings.SAMPLING_RATE not in (1, 10, 100, 1000):
        raise ValueError("Unsupported sampling rate.")

    # Check analog channel indices.
    if (
        not settings.ANALOG_CHANNELS
        or any(channel not in range(6)
               for channel in settings.ANALOG_CHANNELS)
        or len(set(settings.ANALOG_CHANNELS))
        != len(settings.ANALOG_CHANNELS)
    ):
        raise ValueError(
            "Analog channels must be unique indices from 0 to 5."
        )

    if settings.BATCH_SAMPLES <= 0:
        raise ValueError("BATCH_SAMPLES must be positive.")

    if settings.WINDOW_SECONDS <= 0:
        raise ValueError("WINDOW_SECONDS must be positive.")

    if settings.INFERENCE_STRIDE_MS <= 0:
        raise ValueError(
            "INFERENCE_STRIDE_MS must be positive."
        )

    if not 0 <= settings.BLINK_THRESHOLD <= 1:
        raise ValueError(
            "BLINK_THRESHOLD must be between 0 and 1."
        )


validate_settings()