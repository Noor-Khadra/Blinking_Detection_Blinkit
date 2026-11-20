"""Turn detected blink times into deliberate commands (author: Rahma Bezzaouia).

The EEG detector reports *every* blink it finds. Most blinks are spontaneous,
so one blink must never trigger anything. A command is a short burst of
blinks that are close together and followed by a pause:

    blink  blink                -> 2 blinks  -> NEXT track
    blink  blink  blink         -> 3 blinks  -> PREVIOUS track
    blink  blink  blink  blink  -> 4 blinks  -> PLAY / PAUSE

Rules (all in seconds of EEG time, i.e. the detector's `peak_s`):
  * consecutive blinks in one burst are at most `max_gap` apart;
    the detector already refuses alerts closer than 0.35 s, so the useful
    range of `max_gap` is about 0.5-1.0 s;
  * a burst ends when no new blink arrives within `max_gap`;
  * after a command, the next `cooldown` seconds are ignored, so a long burst
    is not read twice.

The recognizer only needs timestamps, so the same code runs on the live
detector, on saved alert files, on webcam labels and on simulations.
"""
from dataclasses import dataclass, field

COMMANDS = {2: "NEXT", 3: "PREVIOUS", 4: "PLAY_PAUSE"}


@dataclass
class GestureRecognizer:
    max_gap: float = 0.8
    cooldown: float = 1.0
    commands: dict = field(default_factory=lambda: dict(COMMANDS))
    _burst: list = field(default_factory=list)
    _blocked_until: float = -1e9

    def push(self, t):
        """Add one blink time. Returns a finished command (dict) or None."""
        result = self.flush(t)
        if t < self._blocked_until:
            return result
        self._burst.append(t)
        return result

    def flush(self, now):
        """Close the burst if `now` is more than `max_gap` after its last blink."""
        if self._burst and now - self._burst[-1] > self.max_gap:
            burst, self._burst = self._burst, []
            command = self.commands.get(len(burst))
            if command is None and len(burst) > max(self.commands):
                command = "IGNORED_TOO_MANY"
            if command:
                decided_at = burst[-1] + self.max_gap
                self._blocked_until = decided_at + self.cooldown
                return {"command": command, "blinks": len(burst), "start_s": burst[0],
                        "end_s": burst[-1], "decided_s": decided_at}
        return None


def recognize(times, **kwargs):
    """Run the recognizer over a whole list of blink times (offline use)."""
    rec = GestureRecognizer(**kwargs)
    out = []
    for t in sorted(times):
        r = rec.push(t)
        if r:
            out.append(r)
    r = rec.flush(float("inf"))
    if r:
        out.append(r)
    return [c for c in out if c["command"] != "IGNORED_TOO_MANY"]
