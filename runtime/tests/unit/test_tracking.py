import pytest
from conftest import det
from ppe_runtime.tracking import create_tracker
from ppe_runtime.utils.config import TrackingConfig


def walk(tracker, frames=10, image=None):
    out = []
    for f in range(frames):
        dets = [
            det("person", (10 + 3 * f, 10, 90 + 3 * f, 200)),
            det("person", (300 - 3 * f, 20, 380 - 3 * f, 210)),
            det("helmet", (30 + 3 * f, 10, 60 + 3 * f, 30)),
        ]
        out.append(tracker.update(dets, image))
    return out


@pytest.mark.parametrize("kind", ["bytetrack", "botsort"])
def test_ids_are_stable_and_only_people_tracked(kind, image):
    cfg = TrackingConfig(
        tracker_type=kind,
        **(
            {
                "gmc_method": "none",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.8,
                "with_reid": False,
                "model": "auto",
            }
            if kind == "botsort"
            else {}
        ),
    )
    tracker = create_tracker(cfg, {"person"})
    frames = walk(tracker, image=image)
    last = frames[-1]
    people = sorted((d for d in last if d.class_name == "person"), key=lambda d: d.box[0])
    assert len(people) == 2 and all(p.track_id is not None for p in people)
    assert people[0].track_id != people[1].track_id
    assert all(d.track_id is None for d in last if d.class_name == "helmet")
    # tracked boxes stay xyxy and close to the input (Kalman-smoothed)
    assert people[0].box == pytest.approx((37, 10, 117, 200), abs=3)
    assert people[1].box == pytest.approx((273, 20, 353, 210), abs=3)
    # the left-moving person keeps the id it had a few frames earlier
    earlier = sorted((d for d in frames[3] if d.class_name == "person"), key=lambda d: d.box[0])
    assert earlier[0].track_id == people[0].track_id


def test_reset_restarts_state(image):
    tracker = create_tracker(TrackingConfig(), {"person"})
    walk(tracker, 3)
    tracker.reset()
    assert tracker.update([], image) == []


def test_disabled():
    assert create_tracker(None) is None


def test_botsort_auto_reid_rejected():
    cfg = TrackingConfig(
        tracker_type="botsort",
        with_reid=True,
        model="auto",
        gmc_method="none",
        proximity_thresh=0.5,
        appearance_thresh=0.8,
    )
    with pytest.raises(ValueError, match="ReID"):
        create_tracker(cfg)
