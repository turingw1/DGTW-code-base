from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TeacherConfigView:
    type: str = "none"
    name_or_path: str | None = None
    target_type: str | None = None


class NullTeacher:
    def is_enabled(self) -> bool:
        return False

    def prepare(self, device: Any) -> None:
        return None
