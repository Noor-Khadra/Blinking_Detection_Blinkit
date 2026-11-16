# Rana 1 auxiliary EEG windows

`rana1` is a local session alias for a public EEGEyeNet recording. The
source participant is `eegeyenet_ep10`. See
[the recording provenance](../../derived/rana/session1/provenance.json)
for the source, recording device and conversion.

This folder contains single-channel EEG windows, class
targets, event annotations and the row index in BlinkIt's model-data
schema. There are 56 blink and 221 no-blink windows. The webcam-format
rows carry eye-tracker events and virtual frame numbers.

The continuous EEG pilot can read this package as an auxiliary training
source. The published Noor model inputs, evaluation recordings and
splits remain independent of these examples.

```sh
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak \
  --models waveform_svm svm_cnn_ensemble cnn_polarity_normalized \
  --auxiliary-windows data/model/rana1 --output reports/blink_external/pooled
```

Use `--auxiliary-fraction 0.5` to train with a reproducible subset of
28 imported blink windows. No-blink windows are sampled at the same
3:1 ratio within every training session.

Use `--match-auxiliary-gain` to estimate a single external gain from
the Noor recordings in the current training fold. See the
[comparison results](../../../reports/blink_external/README.md).
