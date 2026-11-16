import asyncio
import logging
import time

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from acquisition import BitalinoAcquisition


# Configure logging.
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


# Locate the frontend files.
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# Create the FastAPI application.
app = FastAPI(title="BITalino Blink Detection")


# Make CSS and JavaScript files accessible.
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# Serve the main webpage.
@app.get("/")
async def home():

    return FileResponse(STATIC_DIR / "index.html")


# WebSocket endpoint for real-time EEG acquisition.
@app.websocket("/ws")
async def stream_eeg(websocket: WebSocket):

    await websocket.accept()

    # Create an acquisition object for this connection.
    device = BitalinoAcquisition(
        mac_address=settings.BITALINO_MAC_ADDRESS,
        sampling_rate=settings.SAMPLING_RATE,
        channels=list(settings.ANALOG_CHANNELS),
        batch_samples=settings.BATCH_SAMPLES
    )

    try:

        # Connect and start acquisition without blocking
        # FastAPI's asynchronous event loop.
        await asyncio.to_thread(device.connect)

        await websocket.send_json({
            "type": "status",
            "message": "BITalino connected. Streaming started."
        })

        while True:

            # Hardware reading is a blocking operation.
            # Run it in a worker thread.
            samples = await asyncio.to_thread(
                device.read_batch
            )

            # Time when the batch was received by the computer.
            received_at = time.monotonic()

            # Convert NumPy values into ordinary Python values
            # so they can be serialized as JSON.
            await websocket.send_json({
                "type": "eeg_batch",
                "sampling_rate": settings.SAMPLING_RATE,
                "channel": "A2",
                "samples": samples[:, 0].tolist(),
                "received_at": received_at
            })

    except WebSocketDisconnect:

        log.info("Browser disconnected.")

    except Exception as error:

        log.exception("EEG streaming error.")

        try:
            await websocket.send_json({
                "type": "error",
                "message": str(error)
            })
        except Exception:
            pass

    finally:

        # Stop acquisition and release the Bluetooth connection.
        try:
            await asyncio.to_thread(device.close)
        except Exception:
            log.exception("Error while closing BITalino.")

        log.info("Streaming session ended.")


# Run the application locally.
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )