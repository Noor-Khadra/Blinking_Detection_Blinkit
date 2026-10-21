# Blinking_Detection_Blinkit
Real-time eye-blink detection from single-channel BITalino EEG, trained using webcam-labeled blinks as ground truth.

## How it works
1. SENSE    - BITalino EEG (3 electrodes, 1 channel) + webcam recorded together
2. PROCESS  - Video blinks are detected and checked, then used to label the
              EEG; a model learns the blink pattern from the labeled signal
3. SHOW     - Live EEG with detected blinks marked

## Hardware
- BITalino (EEG, 3 electrodes, 1 channel)
- Webcam (training/labeling only)
- GCO308 usb-plug camera (training/labeling only)

## Repository structure

## Installation

## Team
Rana Ezzeddine
Rahma Bezzaouia
Noor Khadra
Katrina Issa

