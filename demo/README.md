A few important decisions are built into this version:
- It reads only physical channel A4.
- It uses the BITalino device's Bluetooth MAC address, not the computer's Bluetooth Network Connection adapter address.
- It sends batches of EEG samples to the browser through a WebSocket.
- It does not assume the blink model's input/output format yet. Model inference will be added once we know the model's actual interface.
- The sampling rate must match the rate expected by the model, or the signal must be resampled before inference.