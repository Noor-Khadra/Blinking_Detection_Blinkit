from config import settings

print("Sampling rate:", settings.SAMPLING_RATE)
print("Analog channels:", settings.ANALOG_CHANNELS)
print("Batch samples:", settings.BATCH_SAMPLES)
print("Window duration:", settings.WINDOW_SECONDS)
print("Inference stride:", settings.INFERENCE_STRIDE_MS)
print("Blink threshold:", settings.BLINK_THRESHOLD)
print("Model module:", settings.MODEL_MODULE)