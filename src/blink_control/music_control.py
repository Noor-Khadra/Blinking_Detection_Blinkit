"""BlinkIT Music Control: blink gestures -> music commands (author: Rahma Bezzaouia).

The team's live app (src/live_blink/app.py) already detects blinks from EEG and
writes every detection to data/raw/live_blinks/live/<session>/blinks.csv.
This window reads those detections as they arrive, groups them into gestures
(see gestures.py) and turns them into music commands:

    2 blinks -> NEXT      3 blinks -> PREVIOUS      4 blinks -> PLAY / PAUSE

Nothing in the team's code is changed: run their app as usual, then run this.

Blink sources
  --source live        newest live session of the team app (default)
  --source keyboard    press SPACE for each blink (rehearsal without EEG)
  --source replay FILE replay a saved blinks.csv in real time

Options
  --media-keys         also send real media keys to the computer (controls
                       Spotify / YouTube / Apple Music). Needs: pip install pynput
  --trial N            measurement mode: asks for N random gestures, records
                       whether each was recognized, and counts commands that
                       happen when no gesture was asked for. Saves results to
                       reports/blink_control/trials/.
  --max-gap S          max seconds between blinks in one gesture (default 0.8)

Examples (from the repo root)
  python3 src/blink_control/music_control.py --source keyboard
  python3 src/blink_control/music_control.py --source live --media-keys
  python3 src/blink_control/music_control.py --source live --trial 30
"""
import argparse
import csv
import json
import random
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gestures import COMMANDS, GestureRecognizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "data/raw/live_blinks/live"
TRIAL_DIR = ROOT / "reports/blink_control/trials"
PLAYLIST = ["Track 1", "Track 2", "Track 3", "Track 4", "Track 5"]
BG, CARD, TEXT, MUTED, ACCENT, GOOD, BAD = "#0f172a", "#1e293b", "#f1f5f9", "#94a3b8", "#38bdf8", "#34d399", "#f87171"


class MediaKeys:
    def __init__(self, enabled):
        self.keyboard = None
        if enabled:
            try:
                from pynput.keyboard import Controller, Key
                self.keyboard, self.Key = Controller(), Key
            except Exception as exc:  # pynput missing or no permission
                print(f"Media keys disabled: {exc}. Install with: pip install pynput")

    def send(self, command):
        if not self.keyboard:
            return
        key = {"NEXT": self.Key.media_next, "PREVIOUS": self.Key.media_previous,
               "PLAY_PAUSE": self.Key.media_play_pause}[command]
        self.keyboard.press(key)
        self.keyboard.release(key)


class LiveTail:
    """Follow the newest blinks.csv written by the team's live app."""

    def __init__(self, folder=LIVE_DIR, after=None):
        self.folder, self.after = folder, after or time.time()
        self.path, self.handle, self.header = None, None, None

    def _newest(self):
        files = [p for p in self.folder.glob("*/blinks.csv") if p.stat().st_mtime >= self.after - 5]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def poll(self):
        newest = self._newest()
        if newest and newest != self.path:
            if self.handle:
                self.handle.close()
            self.path, self.handle, self.header = newest, open(newest, newline=""), None
        if not self.handle:
            return []
        rows = []
        while True:
            pos = self.handle.tell()
            line = self.handle.readline()
            if not line or not line.endswith("\n"):
                self.handle.seek(pos)
                break
            values = next(csv.reader([line]))
            if self.header is None:
                self.header = values
                continue
            rows.append(dict(zip(self.header, values)))
        now = time.monotonic()
        # place each blink at its real time: arrival minus the detector's delay
        return [now - float(r.get("algorithm_delay_s") or 0) for r in rows]

    def status(self):
        return f"reading {self.path.parent.name}" if self.path else "waiting for the live app to start…"


class ReplaySource:
    def __init__(self, path):
        with open(path, newline="") as f:
            self.times = sorted(float(r["peak_s"]) for r in csv.DictReader(f))
        self.start, self.i = time.monotonic(), 0

    def poll(self):
        now, out = time.monotonic() - self.start, []
        while self.i < len(self.times) and self.times[self.i] <= now:
            out.append(self.start + self.times[self.i])
            self.i += 1
        return out

    def status(self):
        return f"replay {self.i}/{len(self.times)} blinks"


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.rec = GestureRecognizer(max_gap=args.max_gap)
        self.keys = MediaKeys(args.media_keys)
        self.track, self.playing, self.blinks = 0, True, 0
        self.pending = []
        self.source = None
        if args.source == "live":
            self.source = LiveTail()
        elif args.source == "replay":
            self.source = ReplaySource(args.file)
        self.trial = None
        if args.trial:
            self.trial = {"todo": [random.choice(list(COMMANDS.values())) for _ in range(args.trial)],
                          "current": None, "asked_at": None, "rows": [], "unprompted": [],
                          "next_prompt": time.monotonic() + 5, "started": time.monotonic()}
        self.build()
        if args.source == "keyboard":
            root.bind("<space>", lambda e: self.blink(time.monotonic()))
        self.tick()

    def build(self):
        r = self.root
        r.title("BlinkIT · Music Control")
        r.configure(bg=BG)
        r.geometry("640x520")
        tk.Label(r, text="BlinkIT · Music Control", font=("Helvetica", 22, "bold"), bg=BG, fg=TEXT).pack(pady=(18, 2))
        tk.Label(r, text="2 blinks = next   ·   3 blinks = previous   ·   4 blinks = play / pause",
                 font=("Helvetica", 13), bg=BG, fg=MUTED).pack()
        card = tk.Frame(r, bg=CARD, padx=20, pady=16)
        card.pack(fill="x", padx=24, pady=16)
        self.now = tk.Label(card, font=("Helvetica", 26, "bold"), bg=CARD, fg=TEXT)
        self.now.pack()
        self.state = tk.Label(card, font=("Helvetica", 15), bg=CARD, fg=ACCENT)
        self.state.pack()
        self.dots = tk.Label(card, font=("Helvetica", 30), bg=CARD, fg=ACCENT)
        self.dots.pack(pady=4)
        self.last = tk.Label(card, font=("Helvetica", 16, "bold"), bg=CARD, fg=GOOD)
        self.last.pack()
        self.prompt = tk.Label(r, font=("Helvetica", 18, "bold"), bg=BG, fg="#fbbf24")
        self.prompt.pack(pady=4)
        self.log = tk.Listbox(r, height=6, bg=CARD, fg=TEXT, font=("Menlo", 11), borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", padx=24)
        self.status = tk.Label(r, font=("Helvetica", 11), bg=BG, fg=MUTED)
        self.status.pack(pady=8)
        self.redraw()

    def redraw(self):
        self.now.config(text=PLAYLIST[self.track])
        self.state.config(text="▶ playing" if self.playing else "⏸ paused")
        self.dots.config(text="● " * len(self.rec._burst) or " ")

    def blink(self, t):
        self.blinks += 1
        self.handle(self.rec.push(t))
        self.redraw()

    def handle(self, result):
        if not result:
            return
        cmd = result["command"]
        if cmd == "NEXT":
            self.track = (self.track + 1) % len(PLAYLIST)
        elif cmd == "PREVIOUS":
            self.track = (self.track - 1) % len(PLAYLIST)
        elif cmd == "PLAY_PAUSE":
            self.playing = not self.playing
        self.keys.send(cmd)
        self.last.config(text=f"{cmd.replace('_', ' / ')}  ({result['blinks']} blinks)")
        self.log.insert(0, f"{datetime.now():%H:%M:%S}  {cmd:<11} {result['blinks']} blinks")
        if self.trial:
            self.trial_record(cmd)
        self.redraw()

    # ---- measurement mode -------------------------------------------------
    def trial_record(self, cmd):
        t = self.trial
        if t["current"] and time.monotonic() - t["asked_at"] <= 6:
            t["rows"].append({"asked": t["current"], "recognized": cmd,
                              "latency_s": round(time.monotonic() - t["asked_at"], 2)})
            t["current"] = None
            t["next_prompt"] = time.monotonic() + random.uniform(6, 9)
        else:
            t["unprompted"].append({"command": cmd, "at_s": round(time.monotonic() - t["started"], 1)})

    def trial_tick(self):
        t, now = self.trial, time.monotonic()
        if t["current"] and now - t["asked_at"] > 6:
            t["rows"].append({"asked": t["current"], "recognized": None, "latency_s": None})
            t["current"], t["next_prompt"] = None, now + random.uniform(6, 9)
        if not t["current"] and t["todo"] and now >= t["next_prompt"]:
            t["current"], t["asked_at"] = t["todo"].pop(), now
        if t["current"]:
            n = [k for k, v in COMMANDS.items() if v == t["current"]][0]
            self.prompt.config(text=f"Now: {t['current'].replace('_', ' / ')}  → blink {n}×")
        elif t["todo"]:
            self.prompt.config(text="relax and blink normally…")
        else:
            self.prompt.config(text="Trial finished — results saved")
            self.save_trial()
            self.trial = None

    def save_trial(self):
        t = self.trial
        TRIAL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        minutes = (time.monotonic() - t["started"]) / 60
        summary = {}
        for c in COMMANDS.values():
            asked = [r for r in t["rows"] if r["asked"] == c]
            summary[c] = {"asked": len(asked),
                          "correct": sum(r["recognized"] == c for r in asked),
                          "wrong": sum(r["recognized"] not in (None, c) for r in asked),
                          "missed": sum(r["recognized"] is None for r in asked)}
        out = {"source": self.args.source, "max_gap_s": self.args.max_gap, "minutes": round(minutes, 2),
               "summary": summary, "unprompted_commands": len(t["unprompted"]),
               "unprompted_per_10min": round(10 * len(t["unprompted"]) / max(minutes, 1e-9), 2),
               "attempts": t["rows"], "unprompted": t["unprompted"]}
        path = TRIAL_DIR / f"trial_{stamp}.json"
        path.write_text(json.dumps(out, indent=2))
        self.log.insert(0, f"saved {path.name}")
        print(json.dumps(summary, indent=2))

    # ---- main loop --------------------------------------------------------
    def tick(self):
        if self.source:
            for t in self.source.poll():
                self.blink(t)
        self.handle(self.rec.flush(time.monotonic()))
        if self.trial:
            self.trial_tick()
        src = self.source.status() if self.source else "keyboard: press SPACE for each blink"
        self.status.config(text=f"{src}   ·   {self.blinks} blinks seen   ·   max gap {self.args.max_gap}s"
                                + ("   ·   media keys ON" if self.keys.keyboard else ""))
        self.redraw()
        self.root.after(50, self.tick)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["live", "keyboard", "replay"], default="live")
    p.add_argument("file", nargs="?", help="blinks.csv for --source replay")
    p.add_argument("--media-keys", action="store_true")
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--max-gap", type=float, default=0.8)
    args = p.parse_args()
    if args.source == "replay" and not args.file:
        p.error("--source replay needs a blinks.csv file")
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
