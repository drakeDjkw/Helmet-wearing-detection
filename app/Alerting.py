"""
Alert logic for the realtime page.

A single frame is not enough to raise an alert: occlusion and camera angle make
predictions "flicker" between frames. The tracker therefore turns per-frame
VIOLATION results into alert *events*:

    * An event is RAISED when at least `min_hits` of the last `window` processed
      frames are violations, and at least `cooldown_s` seconds have passed since
      the previous event ended.
    * An event stays ACTIVE while violations continue, so one worker standing in
      view produces one alert, not one per frame.
    * An event is CLEARED when none of the last `window` processed frames is a
      violation.

Each event records its alert latency: the time from the first violating frame
in the window that raised it to the moment it was raised.

This module has no Streamlit or model imports, so it can be tested on its own.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Deque, List, Optional, Tuple

import io
import math
import struct
import wave

RAISED = "raised"
CLEARED = "cleared"


@dataclass
class AlertEvent:
    id: int
    first_seen_ts: float          # first violating frame in the raising window
    raised_ts: float              # when the alert was raised
    ended_ts: Optional[float] = None
    max_without: int = 0          # most heads without helmet in one frame
    max_conf: float = 0.0         # highest Head_without_helmet confidence
    frames: int = 0               # violating frames during the event
    snapshot: Optional[object] = field(default=None, repr=False)  # annotated BGR frame

    @property
    def active(self) -> bool:
        return self.ended_ts is None

    @property
    def latency_s(self) -> float:
        return self.raised_ts - self.first_seen_ts

    def duration_s(self, now: float) -> float:
        return (self.ended_ts if self.ended_ts is not None else now) - self.raised_ts

    def as_row(self, now: float, time_fmt: Optional[Callable[[float], str]] = None) -> dict:
        """One log row. `time_fmt` turns a timestamp into text (default: wall-clock HH:MM:SS)."""
        fmt = time_fmt or (lambda ts: datetime.fromtimestamp(ts).strftime("%H:%M:%S"))
        return {
            "event": self.id,
            "raised": fmt(self.raised_ts),
            "ended": fmt(self.ended_ts) if self.ended_ts is not None else "active",
            "duration_s": round(self.duration_s(now), 1),
            "alert_latency_s": round(self.latency_s, 2),
            "max_heads_without": self.max_without,
            "max_confidence": round(self.max_conf, 3),
            "violating_frames": self.frames,
        }


class AlertTracker:
    """Debounces per-frame violations into alert events (not thread-safe by itself)."""

    def __init__(self, window: int = 5, min_hits: int = 3, cooldown_s: float = 10.0,
                 max_events: int = 200, max_snapshots: int = 20):
        self.max_events = max_events
        self.max_snapshots = max_snapshots
        self.events: List[AlertEvent] = []
        self._history: Deque[Tuple[float, bool]] = deque()
        self._last_end_ts = -math.inf
        self.configure(window, min_hits, cooldown_s)

    def configure(self, window: int, min_hits: int, cooldown_s: float) -> None:
        window = max(1, int(window))
        self.window = window
        self.min_hits = max(1, min(int(min_hits), window))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._history = deque(self._history, maxlen=window)

    @property
    def current(self) -> Optional[AlertEvent]:
        if self.events and self.events[-1].active:
            return self.events[-1]
        return None

    def reset(self) -> None:
        self.events.clear()
        self._history.clear()
        self._last_end_ts = -math.inf

    def update(self, ts: float, is_violation: bool, n_without: int = 0,
               max_conf: float = 0.0, snapshot=None) -> Optional[str]:
        """Feed one processed frame. Returns RAISED, CLEARED or None."""
        self._history.append((ts, is_violation))
        ev = self.current

        if ev is not None:
            if is_violation:
                ev.frames += 1
                ev.max_without = max(ev.max_without, n_without)
                ev.max_conf = max(ev.max_conf, max_conf)
            full = len(self._history) == self.window
            if full and not any(v for _, v in self._history):
                ev.ended_ts = ts
                self._last_end_ts = ts
                return CLEARED
            return None

        hits = [t for t, v in self._history if v]
        if len(hits) >= self.min_hits and ts - self._last_end_ts >= self.cooldown_s:
            ev = AlertEvent(id=(self.events[-1].id + 1) if self.events else 1,
                            first_seen_ts=hits[0], raised_ts=ts,
                            max_without=n_without, max_conf=max_conf,
                            frames=len(hits), snapshot=snapshot)
            self.events.append(ev)
            self._trim()
            return RAISED
        return None

    def rows(self, now: float, time_fmt: Optional[Callable[[float], str]] = None) -> List[dict]:
        return [e.as_row(now, time_fmt) for e in self.events]

    def close(self, ts: float) -> None:
        """End an alert that is still active, e.g. when a video ends."""
        ev = self.current
        if ev is not None:
            ev.ended_ts = ts
            self._last_end_ts = ts

    def trim_snapshots(self) -> None:
        """Keep snapshots only for the newest `max_snapshots` events (bounds memory)."""
        with_snap = [e for e in self.events if e.snapshot is not None]
        for e in with_snap[: max(0, len(with_snap) - self.max_snapshots)]:
            e.snapshot = None

    def _trim(self) -> None:
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]
        with_snap = [e for e in self.events if e.snapshot is not None]
        for e in with_snap[: max(0, len(with_snap) - self.max_snapshots)]:
            e.snapshot = None


def beep_wav(variant: int = 0, freq: float = 880.0, beeps: int = 3,
             beep_s: float = 0.15, gap_s: float = 0.1, rate: int = 16000) -> bytes:
    """A short alarm sound as WAV bytes.

    `variant` adds a few samples of silence, so each alert produces different bytes
    and the browser plays it again instead of treating it as the same file.
    """
    frames = bytearray()
    for b in range(beeps):
        for i in range(int(beep_s * rate)):
            env = min(1.0, i / 200, (beep_s * rate - i) / 200)   # avoid clicks
            frames += struct.pack("<h", int(12000 * env * math.sin(2 * math.pi * freq * i / rate)))
        if b < beeps - 1:
            frames += b"\x00\x00" * int(gap_s * rate)
    frames += b"\x00\x00" * (variant % 50)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()