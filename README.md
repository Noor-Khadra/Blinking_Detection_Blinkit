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

project/
├── data/
│   ├── raw/                      
│   │   └── 25-10-2026/
│   │       ├── eeg_opensignals.txt
│   │       ├── eeg.csv          
│   │       ├── webcam.mp4
│   │       ├── webcam_timestamps.csv
│   │       ├── eyecam.mp4
│   │       ├── eyecam_timestamps.csv
│   │       └── meta.json          
│   └── derived/
│       └── 
├── models/                        
├── reports/                       
└── src/
    ├── recording_code/ 

## Installation

## Team
Rana Ezzeddine
Rahma Bezzaouia
Noor Khadra
Katrina Issa


## Webcam labeling (Rana)

See [the webcam workflow](src/webcam/README.md) for extraction, candidate generation,
manual review, timing limitations, and the handoff to EEG modeling.
