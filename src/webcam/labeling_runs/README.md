# Scripts used for the review runs

The candidate script in this directory uses the 80th-percentile open-eye reference, per-eye EAR thresholds, and MediaPipe blink scores used to generate candidates for the reviewed Noor recordings. The configurable pipeline in the parent directory is the branch's main entry point.

For Noor 1, the original extraction used MediaPipe 1.1.1. Noor 2 used the extractor in this directory with the pinned 0.10.21 environment. Noor 3 used the parent directory's PyAV extractor with MediaPipe 0.10.21. Obtain the Face Landmarker task model from the link in the parent README. Use a separate working folder per recording and keep the output filename `eye_measurements.csv` beside a copy of `build_candidates.py`. The candidate script writes the initial proposals, summary, and review data.

Check the source video frame timeline before reusing an annotation. The session CSVs contain human-reviewed events in video playback time.
