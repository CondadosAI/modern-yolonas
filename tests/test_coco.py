"""COCOClass must stay pinned to COCO_NAMES.

The enum is written out by hand so editors can complete it, which means it is a
second copy of an ordering that already exists. These tests are what stop the
two from drifting: a class inserted in one and not the other renumbers every id
after it, which would mislabel detections rather than raise.
"""

from __future__ import annotations

import json

import numpy as np
import supervision as sv

from modern_yolonas import COCO_NAMES, COCOClass


class TestPinnedToCOCONames:
    def test_same_length(self):
        assert len(COCOClass) == len(COCO_NAMES) == 80

    def test_every_member_matches_its_name_and_position(self):
        for member in COCOClass:
            assert COCO_NAMES[member.value] == member.label, (
                f"{member.name} = {member.value} but COCO_NAMES[{member.value}] is {COCO_NAMES[member.value]!r}"
            )

    def test_values_are_a_contiguous_range(self):
        assert [m.value for m in COCOClass] == list(range(len(COCO_NAMES)))

    def test_label_reverses_the_identifier(self):
        assert COCOClass.TRAFFIC_LIGHT.label == "traffic light"
        assert COCOClass.PERSON.label == "person"


class TestBehavesAsAnInt:
    def test_is_an_int(self):
        assert isinstance(COCOClass.PERSON, int)
        assert COCOClass.PERSON == 0

    def test_indexes_coco_names(self):
        assert COCO_NAMES[COCOClass.CAR] == "car"

    def test_json_serialises_as_a_number(self):
        assert json.loads(json.dumps({"class_id": COCOClass.DOG}))["class_id"] == 16

    def test_reverse_lookup(self):
        assert COCOClass(2) is COCOClass.CAR


class TestFiltersDetections:
    """The reason the enum exists: it has to work on a real sv.Detections."""

    @staticmethod
    def _detections(class_ids):
        n = len(class_ids)
        return sv.Detections(
            xyxy=np.tile(np.array([0, 0, 1, 1], dtype=np.float32), (n, 1)),
            confidence=np.full(n, 0.9, dtype=np.float32),
            class_id=np.asarray(class_ids),
        )

    def test_equality_filter(self):
        detections = self._detections([0, 2, 0, 16])
        people = detections[detections.class_id == COCOClass.PERSON]
        assert len(people) == 2

    def test_membership_filter(self):
        detections = self._detections([0, 2, 7, 16])
        vehicles = detections[np.isin(detections.class_id, [COCOClass.CAR, COCOClass.TRUCK])]
        assert len(vehicles) == 2

    def test_filter_is_identical_to_the_bare_integer(self):
        detections = self._detections([0, 2, 0, 16])
        by_enum = detections[detections.class_id == COCOClass.PERSON]
        by_int = detections[detections.class_id == 0]
        assert np.array_equal(by_enum.class_id, by_int.class_id)
