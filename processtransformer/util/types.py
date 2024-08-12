

import typing as tp

Event = str
XEvent = str
YEvent = str

Trace = tp.List[Event]
SimpleSoftmax = tp.List[YEvent]
SoftmaxVector = tp.List[tp.Tuple[YEvent, float]]
AttentionVector = tp.List[tp.Tuple[XEvent, float]]

XWordDict = tp.Dict[XEvent, int]
YWordDict = tp.Dict[YEvent, int]

LocalScoreDict = tp.Dict[YEvent, tp.Dict[XEvent, tp.List[float]]]
GlobalScoreDict = tp.Dict[YEvent, tp.Dict[XEvent, float]]

GeneralList = tp.List[tp.Tuple[tp.Any, tp.Any]]
EventValueList = tp.List[tp.Tuple[Event, float]]
EventValueDict = tp.Dict[Event, float]

RelationsDict = tp.Dict[tp.Tuple[XEvent], tp.Tuple[YEvent, ...]]
