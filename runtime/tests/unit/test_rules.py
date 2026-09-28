from conftest import det

from ppe_runtime.ppe import PersonStatus, RuleEngine


def status(track_id, missing=(), worn=()):
    return PersonStatus(det("person", (0, 0, 100, 200), track_id=track_id), frozenset(worn), frozenset(missing))


def test_raises_once_after_min_frames(rules):
    engine = RuleEngine(rules)  # min_frames=3
    out = [engine.evaluate([status(1, {"vest"})]) for _ in range(6)]
    assert [len(v) for v in out] == [0, 0, 1, 0, 0, 0]
    v = out[2][0]
    assert v.track_id == 1 and v.missing == {"vest"} and v.frames == 3 and v.severity == "WARNING"


def test_compliance_rearms(rules):
    engine = RuleEngine(rules)
    seq = [{"vest"}] * 3 + [set()] + [{"vest"}] * 3
    raised = [len(engine.evaluate([status(1, m)])) for m in seq]
    assert raised == [0, 0, 1, 0, 0, 0, 1]


def test_missing_set_change_restarts_streak(rules):
    engine = RuleEngine(rules)
    seq = [{"vest"}, {"vest"}, {"vest", "helmet"}, {"vest", "helmet"}, {"vest", "helmet"}]
    raised = [engine.evaluate([status(1, m)]) for m in seq]
    assert [len(r) for r in raised] == [0, 0, 0, 0, 1]
    assert raised[-1][0].severity == "CRITICAL"  # highest of helmet=CRITICAL, vest=WARNING


def test_tracks_are_independent(rules):
    engine = RuleEngine(rules)
    for _ in range(2):
        engine.evaluate([status(1, {"vest"}), status(2, {"vest"})])
    out = engine.evaluate([status(1, {"vest"}), status(2, set())])
    assert [v.track_id for v in out] == [1]


def test_untracked_only_when_min_frames_is_one(rules):
    assert RuleEngine(rules).evaluate([status(None, {"vest"})]) == []
    engine = RuleEngine(rules.model_copy(update={"min_frames": 1}))
    assert len(engine.evaluate([status(None, {"vest"})])) == 1
    assert len(engine.evaluate([status(None, {"vest"})])) == 1  # no debounce without identity


def test_stale_tracks_forgotten(rules):
    engine = RuleEngine(rules)  # forget_after=10
    engine.evaluate([status(1, {"vest"})])
    for _ in range(11):
        engine.evaluate([])
    assert engine.active_tracks == 0
