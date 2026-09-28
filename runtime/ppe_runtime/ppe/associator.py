"""Attach PPE items (and negative classes such as `no_helmet`) to the person they belong to."""

from __future__ import annotations

from ppe_runtime.pipeline.types import Detection
from ppe_runtime.ppe.models import PersonStatus
from ppe_runtime.utils.config import RulesConfig
from ppe_runtime.utils.geometry import Box, center_inside, iou, overlap_fraction


class Associator:
    """Each item goes to the single person it matches best; a person then lacks every required item not assigned.

    Methods: `overlap` (share of the item box inside the person box — robust for small items like helmets),
    `iou`, or `center` (item center inside person box; ties broken by overlap).
    """

    def __init__(self, rules: RulesConfig) -> None:
        self.method = rules.association.method
        self.min_score = rules.association.min_score
        self.person_classes = frozenset(rules.person_classes)
        self.required = frozenset(rules.required)
        self.negatives = dict(rules.negatives)
        self._item_of = {name: name for name in rules.required}
        for item, names in rules.aliases.items():
            if item in self.required:
                self._item_of.update(dict.fromkeys(names, item))

    def item_of(self, class_name: str) -> str | None:
        return self._item_of.get(class_name)

    def score(self, item: Box, person: Box) -> float:
        if self.method == "iou":
            return iou(item, person)
        if self.method == "center":
            return 1.0 + overlap_fraction(item, person) if center_inside(item, person) else 0.0
        return overlap_fraction(item, person)

    def _threshold(self) -> float:
        return 1.0 if self.method == "center" else self.min_score

    def associate(self, detections: list[Detection]) -> list[PersonStatus]:
        persons = [d for d in detections if d.class_name in self.person_classes]
        if not persons:
            return []
        worn: list[set[str]] = [set() for _ in persons]
        flagged: list[set[str]] = [set() for _ in persons]
        threshold = self._threshold()
        for det in detections:
            item = self.item_of(det.class_name)
            negative = self.negatives.get(det.class_name)
            if item is None and negative is None:
                continue
            scores = [self.score(det.box, p.box) for p in persons]
            best = max(range(len(persons)), key=scores.__getitem__)
            if scores[best] < threshold:
                continue
            if item is not None:
                worn[best].add(item)
            else:
                flagged[best].add(negative)
        return [
            PersonStatus(p, frozenset(w), frozenset((self.required - w) | f))
            for p, w, f in zip(persons, worn, flagged)
        ]
