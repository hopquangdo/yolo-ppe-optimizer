"""Turn per-frame person status into debounced violations."""

from __future__ import annotations

from dataclasses import dataclass

from ppe_runtime.ppe.constants import SEVERITY_ORDER
from ppe_runtime.ppe.models import PersonStatus, Violation
from ppe_runtime.utils.config import RulesConfig


@dataclass
class _TrackState:
    missing: frozenset[str] = frozenset()
    streak: int = 0
    raised: bool = False
    last_seen: int = 0


class RuleEngine:
    """Raises a violation once a tracked person has missed the same PPE for `min_frames` consecutive frames.

    It fires once per (track, missing set) and re-arms when the person becomes compliant or the missing set changes.
    Untracked people (no `track_id`, i.e. tracking off) cannot be debounced: they raise on every non-compliant frame,
    and only when `min_frames == 1`.
    """

    def __init__(self, rules: RulesConfig) -> None:
        self.min_frames = rules.min_frames
        self.forget_after = rules.forget_after
        self.severity = dict(rules.severity)
        self.default_severity = rules.default_severity
        self._tracks: dict[int, _TrackState] = {}
        self._frame = 0

    def severity_of(self, missing: frozenset[str]) -> str:
        levels = [self.severity.get(item, self.default_severity) for item in missing] or [self.default_severity]
        return max(levels, key=SEVERITY_ORDER.index)

    def evaluate(self, statuses: list[PersonStatus], frame_index: int | None = None) -> list[Violation]:
        self._frame += 1
        frame_index = self._frame if frame_index is None else frame_index
        raised = []
        for status in statuses:
            person, missing = status.person, status.missing
            if person.track_id is None:
                if missing and self.min_frames == 1:
                    raised.append(self._violation(status, 1, frame_index))
                continue
            state = self._tracks.setdefault(person.track_id, _TrackState())
            state.last_seen = self._frame
            if missing != state.missing:
                state.missing, state.streak, state.raised = missing, 0, False
            if not missing:
                continue
            state.streak += 1
            if state.streak >= self.min_frames and not state.raised:
                state.raised = True
                raised.append(self._violation(status, state.streak, frame_index))
        self._forget()
        return raised

    def _violation(self, status: PersonStatus, frames: int, frame_index: int) -> Violation:
        p = status.person
        return Violation(
            p.track_id, status.missing, self.severity_of(status.missing), p.box, p.confidence, frames, frame_index
        )

    def _forget(self) -> None:
        for track_id in [t for t, s in self._tracks.items() if self._frame - s.last_seen > self.forget_after]:
            del self._tracks[track_id]

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._frame = 0
