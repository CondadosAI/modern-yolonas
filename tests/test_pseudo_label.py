"""The class mapping behind every pseudo-label.

The teacher emits contiguous indices 0-79 and the mapping to COCO's own category
ids is positional. If a checkpoint ever reorders its classes, every box in a
123k-image file gets the wrong label, consistently, and the resulting model
trains perfectly well while detecting the wrong things. Nothing raises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from pseudo_label import KNOWN_SYNONYMS, check_class_order, load_categories  # noqa: E402

from modern_yolonas import COCO_NAMES  # noqa: E402


def as_id2label(names: list[str]) -> dict[int, str]:
    return dict(enumerate(names))


class TestClassOrder:
    def test_our_own_names_are_accepted(self):
        check_class_order(as_id2label(list(COCO_NAMES)))

    def test_the_voc_spellings_d_fine_uses_are_accepted(self):
        names = [KNOWN_SYNONYMS.get(n, n) for n in COCO_NAMES]
        assert names != list(COCO_NAMES), "the synonym table should change something"
        check_class_order(as_id2label(names))

    def test_every_synonym_maps_a_real_coco_name(self):
        """A typo in the table would silently stop protecting that class."""
        for ours in KNOWN_SYNONYMS:
            assert ours in COCO_NAMES, f"{ours!r} is not a COCO class name"

    def test_two_swapped_classes_are_rejected(self):
        names = list(COCO_NAMES)
        names[3], names[4] = names[4], names[3]
        with pytest.raises(SystemExit, match="class order mismatch at index 3"):
            check_class_order(as_id2label(names))

    def test_a_shifted_list_is_rejected_at_the_first_difference(self):
        names = list(COCO_NAMES[1:]) + ["extra"]
        with pytest.raises(SystemExit, match="index 0"):
            check_class_order(as_id2label(names))

    def test_the_wrong_number_of_classes_is_rejected(self):
        with pytest.raises(SystemExit, match="79 classes"):
            check_class_order(as_id2label(list(COCO_NAMES[:-1])))


class TestCategories:
    @staticmethod
    def _write(tmp_path: Path, categories: list[dict]) -> Path:
        path = tmp_path / "ann.json"
        path.write_text(json.dumps({"categories": categories}))
        return path

    def test_categories_come_back_sorted_by_coco_id(self, tmp_path):
        """COCO's ids run 1-90 with gaps; the order must be the id order, not file order."""
        categories = [{"id": i, "name": n} for i, n in zip([3, 1, 90], ["c", "a", "z"])]
        result = load_categories(self._write(tmp_path, categories + [
            {"id": i, "name": f"n{i}"} for i in range(10, 87)
        ]))
        assert [c["id"] for c in result] == sorted(c["id"] for c in result)
        assert result[0]["id"] == 1

    def test_a_file_without_eighty_categories_is_rejected(self, tmp_path):
        path = self._write(tmp_path, [{"id": 1, "name": "person"}])
        with pytest.raises(SystemExit, match="1 categories"):
            load_categories(path)
