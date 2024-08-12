
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class FigureData:
    file_path: str | None
    title: str | None
