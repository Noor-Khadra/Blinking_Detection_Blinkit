# Whole-session EEG window comparison
Train on the fit windows of Rana 01, tune epoch, threshold and peak confirmation on its validation minutes, then score all usable intervals of Rana 02. Reverse the sessions and pool event counts. The candidate widths are 0.75, 1.0 and 1.5 seconds. A score-average variant combines the two shorter widths. All use the same EEG filter, 100 ms stride and event matching tolerance.
 
Run `python src/model/experiment_rana_window_length.py` after the streaming comparison. In [summary.csv](summary.csv), compare full-session settings ending in `_f1`. The `_budget` settings use a source-validation false-alert budget. [folds.csv](folds.csv) gives each direction separately. Checkpoints belong to this experiment only.
