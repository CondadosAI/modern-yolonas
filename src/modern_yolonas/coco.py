"""The COCO class ids, by name.

A detection carries a class id, and a bare integer says nothing about what was
found::

    people = detections[detections.class_id == 0]      # 0 is what, exactly?
    people = detections[detections.class_id == COCOClass.PERSON]

``COCOClass`` is an ``IntEnum``, so each member *is* an ``int``: it compares
against ``detections.class_id`` elementwise, indexes ``COCO_NAMES``, and
serialises to JSON as a number. Nothing that accepts a class id needs to know
this type exists.

These ids describe the 80-class COCO taxonomy, so they apply to the pretrained
checkpoints. A model fine-tuned with ``--num-classes 3`` has its own, unrelated
numbering, and these names would quietly mean the wrong thing there.
"""

from __future__ import annotations

from enum import IntEnum


#: Class id marking a COCO ``iscrowd`` region.
#:
#: Crowd annotations mark areas holding many instances that were never separated,
#: so they are neither a trainable target nor background. Dropping them, as this
#: repo used to, teaches the model that a street full of people is background.
#:
#: They travel as ordinary targets carrying this sentinel rather than as a sixth
#: column, which keeps every transform unchanged -- each one treats column 0 as an
#: opaque label and passes it through. The loss splits them out and uses them only
#: to *ignore* the anchors they cover.
CROWD_CLASS = -1


class COCOClass(IntEnum):
    """Class ids of the 80-class COCO taxonomy.

    Members are written out rather than generated so that editors can complete
    them and type checkers can catch a misspelling. ``tests/test_coco.py`` pins
    the list against :data:`~modern_yolonas.inference.visualize.COCO_NAMES`, so
    the two cannot drift apart unnoticed.
    """

    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORCYCLE = 3
    AIRPLANE = 4
    BUS = 5
    TRAIN = 6
    TRUCK = 7
    BOAT = 8
    TRAFFIC_LIGHT = 9
    FIRE_HYDRANT = 10
    STOP_SIGN = 11
    PARKING_METER = 12
    BENCH = 13
    BIRD = 14
    CAT = 15
    DOG = 16
    HORSE = 17
    SHEEP = 18
    COW = 19
    ELEPHANT = 20
    BEAR = 21
    ZEBRA = 22
    GIRAFFE = 23
    BACKPACK = 24
    UMBRELLA = 25
    HANDBAG = 26
    TIE = 27
    SUITCASE = 28
    FRISBEE = 29
    SKIS = 30
    SNOWBOARD = 31
    SPORTS_BALL = 32
    KITE = 33
    BASEBALL_BAT = 34
    BASEBALL_GLOVE = 35
    SKATEBOARD = 36
    SURFBOARD = 37
    TENNIS_RACKET = 38
    BOTTLE = 39
    WINE_GLASS = 40
    CUP = 41
    FORK = 42
    KNIFE = 43
    SPOON = 44
    BOWL = 45
    BANANA = 46
    APPLE = 47
    SANDWICH = 48
    ORANGE = 49
    BROCCOLI = 50
    CARROT = 51
    HOT_DOG = 52
    PIZZA = 53
    DONUT = 54
    CAKE = 55
    CHAIR = 56
    COUCH = 57
    POTTED_PLANT = 58
    BED = 59
    DINING_TABLE = 60
    TOILET = 61
    TV = 62
    LAPTOP = 63
    MOUSE = 64
    REMOTE = 65
    KEYBOARD = 66
    CELL_PHONE = 67
    MICROWAVE = 68
    OVEN = 69
    TOASTER = 70
    SINK = 71
    REFRIGERATOR = 72
    BOOK = 73
    CLOCK = 74
    VASE = 75
    SCISSORS = 76
    TEDDY_BEAR = 77
    HAIR_DRIER = 78
    TOOTHBRUSH = 79

    @property
    def label(self) -> str:
        """The COCO name as it is written on an annotation, e.g. ``"traffic light"``."""
        return self.name.lower().replace("_", " ")
