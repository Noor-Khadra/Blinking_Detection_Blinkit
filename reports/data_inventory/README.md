# BlinkIT data inventory (Rahma Bezzaouia)

This is one table covering every EEG recording in the repo: what it contains,
how its blink labels were made, and how far its timing can be trusted. It ends
with a recommendation for what to train on and what to test on. Every number is
recomputed from files in the repo: `python3 src/data_inventory/build_inventory.py`
→ [`inventory.csv`](inventory.csv), [`sections.csv`](sections.csv).

## All recordings

| Session | Person | Date | EEG min | Clipping | Protocol | Labels | Blinks | Blinks/min | Webcam→EEG timing |
|---|---|---|---:|---:|---|---|---:|---:|---|
| noor1 | Noor | Oct 19 | 11.2 | 4.0% | unscripted | human-reviewed | 62 | 5.5 | **validated**: held-out 35/41 within 150 ms (controls 11–17) |
| noor2 | Noor | Oct 25 | 10.5 | 1.6% | unscripted | human-reviewed | 81 | 7.7 | **validated**: 52/54 (controls 10–16) |
| noor3 | Noor | Nov 11 | 10.9 | 7.1% | unscripted | human-reviewed | 86 | 7.9 | **validated**: 55/57 (controls 12–21) |
| rahma1 | Rahma | Oct 25 | 12.7 | 2.0% | unscripted | human-reviewed, fully re-reviewed Nov 21 | 157 | 12.4 | **not validated**: no capture log, video clock drifts |
| rana_01_clean | Rana | Nov 16 | 10.0 | 3.9% | 8-section script | automatic, unreviewed | 410 | 41.0 | logged at capture (host clocks): 383/410 have an EEG peak within 150 ms |
| rana_02 | Rana | Nov 16 | 10.0 | 3.9% | 8-section script | automatic, unreviewed | 421 | 42.1 | logged: 392/421 |
| noor4-clean | Noor | Nov 16 | 10.0 | 2.1% | 8-section script | automatic, unreviewed | 130 | 13.0 | logged: 123/130 |
| `rana1` (alias) | **public EEGEyeNet participant EP10, not Rana** | — | — | — | public eye-tracking task | eye tracker | 56 | — | from the source dataset (different EEG device) |

**In total:**
- About 75 minutes of our own BITalino EEG from 3 people.
- 386 human-reviewed blinks: Noor 1–3 and Rahma 1.
- 961 automatic, unreviewed blinks: Rana 1–2 and Noor 4.

There is no Katrina recording in the repo.

## What the inventory shows

1. **Noor 1–3 are synchronized.** On blinks the clock was *not* fitted on,
   85–96% line up with an EEG blink within 150 ms, with a median error of
   17–21 ms. Time-shifted controls reach only 19–37%. Of the
   human-reviewed recordings, only Rahma 1 lacks a trustworthy clock.
2. **The best-labeled data is barely used.** Noor 1–3 have 229 human-reviewed,
   synchronized blinks. The current live CNN (`cnn_01`) uses only Rana's two
   sessions. The replay candidate adds Noor 2 only.
3. **The automatic labels look unreliable in some sections.** The scripted
   sessions log which activity happened when. Rana's automatic labels contain
   35–48 "blinks" during the eyes-closed rest and 29–31 during eye movements
   (see [`sections.csv`](sections.csv)). Rana's overall rate, about 41 per
   minute, is 3–7 times Noor's and Rahma's. A webcam eye-openness detector
   counts looking down or squinting as a blink. So some of the "blinks" the
   model learned from are probably eye movements, which would explain false
   alarms. One caveat: the model data marks these sections
   `unscripted_unknown`, so check with Rana that the script timing was
   actually followed before relying on the section boundaries.
4. **Training is dominated by one person, and the test split is within-person.**
   The CNN trains and tests on different *minutes of the same Rana sessions*.
   The team's own unseen-person check (Rana model → Noor 4) drops to 31% F1
   (18% on test blocks), against about 80% on Rana. Cross-person
   generalization is the main weakness, and the current split hides it.
5. **Recordings differ in setup.** EEG channel (A2, A3 or A4), clipping
   (1.6–7.1%) and protocol (unscripted or scripted) all vary between sessions.
   That is a reason to normalize per session and test across people, not a
   reason to discard data.
6. **Some names are misleading.** `rana1` and `data/eeg/raw/rana_recording1.txt`
   are public EEGEyeNet data, not Rana. Rahma's labels sit in
   `data/derived/noor/session4/`.

## Recommended organization

| Role | Data | Why |
|---|---|---|
| **Train** | Noor 1–3 (reviewed), plus Rana 1–2 and Noor 4 with labels in the eyes-closed, eye-movement and artifact sections reviewed or excluded | Uses every reliable blink. Removes the label noise most likely to cause false alarms. |
| **Balance** | Weight each *person* equally, not each window | Stops Rana's ~830 automatic labels from swamping Noor's ~360. |
| **Test** | Leave-one-person-out: train on Rana, test on Noor; train on Noor, test on Rana | The question the project guide asks: does it work on someone it never saw? |
| **Third-person test** | Rahma 1, once its clock is fixed (anchor workflow in `src/rahma_session`) | The only recording from a third person, with 157 reviewed blinks. |
| **Condition tests** | Scripted sections | False alarms per minute during artifacts and eye movements, and recall in the voluntary-blink section (needed for blink commands). |
| **Auxiliary only** | Public EEGEyeNet (`rana1`) | Different device and electrode. |

**Highest-value next step (about 30 minutes, no new recording needed):**
review the automatic labels in the eyes-closed and eye-movement sections of
Rana 1–2 with the same review page used for Rahma 1. Then retrain with
person-balanced weights and report leave-one-person-out results.

**For any future recording:**
- Log capture timestamps.
- Start and end with 3 forceful blinks as sync markers.
- Follow the 8-section script.
- Name files by person and session.
