# BlinkIT Music Control (Rahma Bezzaouia)

This layer turns BlinkIT's blink detections into hands-free commands. It checks
how well the current detector would support that, using the team's own data.

```
BITalino EEG ──► team live app (src/live_blink) ──► blinks.csv ──► gestures.py ──► music command
                 filter + CNN + peak gate           (one row per     2 blinks = NEXT        on-screen player
                                                    detected blink)  3 blinks = PREVIOUS    + optional real
                                                                     4 blinks = PLAY/PAUSE  media keys
```

The team's code is not modified. The controller follows the `blinks.csv` file
that the live app already writes.

## Files

| File | What it does |
|---|---|
| `gestures.py` | Groups blink times into gestures. A single blink never triggers anything. Blinks in a command are at most 0.8 s apart, and a command ends after 0.8 s with no new blink. A 1 s cooldown follows each command. |
| `music_control.py` | Window with a playlist and command log. Sources: live app, keyboard (SPACE = blink) or replay. `--media-keys` controls Spotify, YouTube or Apple Music. `--trial N` measures gesture success live. |
| `evaluate_gestures.py` | Evidence from the existing data. Writes `reports/blink_control/`. |
| `test_gestures.py` | Unit tests. |

## Run

```sh
pip install numpy pandas matplotlib          # analysis
pip install pynput                           # only for --media-keys
python3 src/blink_control/evaluate_gestures.py
python3 -m unittest src/blink_control/test_gestures.py

# rehearsal without EEG
python3 src/blink_control/music_control.py --source keyboard
# live: start the team app (Run_Live_Blinks), then
python3 src/blink_control/music_control.py --source live --media-keys
# live measurement: 30 random prompts, saves reports/blink_control/trials/
python3 src/blink_control/music_control.py --source live --trial 30
```

On macOS, `--media-keys` needs Terminal to be allowed under System Settings →
Privacy & Security → Accessibility.

See [the results](../../reports/blink_control/README.md).
