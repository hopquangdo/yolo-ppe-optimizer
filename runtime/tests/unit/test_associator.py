from conftest import det

from ppe_runtime.ppe import Associator
from ppe_runtime.utils.config import AssociationConfig


def test_items_inside_person_are_worn(rules):
    dets = [det("person", (0, 0, 100, 200)), det("helmet", (30, 0, 70, 30)), det("vest", (10, 60, 90, 130))]
    [status] = Associator(rules).associate(dets)
    assert status.worn == {"helmet", "vest"}
    assert status.compliant


def test_missing_item_and_alias(rules):
    dets = [det("person", (0, 0, 100, 200)), det("hardhat", (30, 0, 70, 30))]
    [status] = Associator(rules).associate(dets)
    assert status.worn == {"helmet"}
    assert status.missing == {"vest"}


def test_item_goes_to_best_overlapping_person_only(rules):
    a, b = det("person", (0, 0, 100, 200)), det("person", (90, 0, 190, 200))
    helmet = det("helmet", (100, 0, 140, 30))  # fully inside b, touches a's edge only
    statuses = Associator(rules).associate([a, b, helmet])
    assert statuses[0].missing == {"helmet", "vest"}
    assert statuses[1].worn == {"helmet"}


def test_item_outside_every_person_is_ignored(rules):
    dets = [det("person", (0, 0, 100, 200)), det("helmet", (300, 300, 340, 330))]
    [status] = Associator(rules).associate(dets)
    assert "helmet" in status.missing


def test_negative_class_flags_item_even_if_worn(rules):
    dets = [det("person", (0, 0, 100, 200)), det("helmet", (30, 0, 70, 30)), det("vest", (10, 60, 90, 130)),
            det("no_helmet", (30, 0, 70, 30))]
    [status] = Associator(rules).associate(dets)
    assert status.missing == {"helmet"}


def test_center_method(rules):
    rules = rules.model_copy(update={"association": AssociationConfig(method="center")})
    person = det("person", (0, 0, 100, 200))
    straddling = det("vest", (60, 60, 160, 130))  # center (110, 95) is outside the person
    [status] = Associator(rules).associate([person, straddling, det("helmet", (40, 0, 60, 20))])
    assert status.missing == {"vest"}


def test_no_persons(rules):
    assert Associator(rules).associate([det("helmet", (0, 0, 10, 10))]) == []
