"""Individual stance artifact. No classification, inference or sentiment logic."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from youtube_pipeline.signal_observations import require_text, require_utc

CLASSIFIED = "CLASSIFIED"
NOT_APPLICABLE = "NOT_APPLICABLE"
UNCERTAIN = "UNCERTAIN"
SKIPPED = "SKIPPED"
ERROR = "ERROR"
CLASSIFICATION_STATUSES = frozenset({CLASSIFIED, NOT_APPLICABLE, UNCERTAIN, SKIPPED, ERROR})


@dataclass(frozen=True, slots=True)
class CommentStance:
    comment_id: str
    author_id: str | None
    event_time_utc: datetime
    proposition_id: str
    stance_category: int | None
    classification_status: str
    quality: str = "passed"

    def __post_init__(self) -> None:
        for name in ("comment_id", "proposition_id", "quality"):
            require_text(name, getattr(self, name))
        require_utc("event_time_utc", self.event_time_utc)
        if self.author_id is not None:
            require_text("author_id", self.author_id)
        if self.classification_status not in CLASSIFICATION_STATUSES:
            raise ValueError("Unsupported classification_status.")
        if self.classification_status == CLASSIFIED:
            if isinstance(self.stance_category, bool) or not isinstance(self.stance_category, int):
                raise TypeError("CLASSIFIED stance_category must be an integer.")
            if self.stance_category not in (1, 2, 3, 4, 5):
                raise ValueError("CLASSIFIED stance_category must be in {1,2,3,4,5}.")
        elif self.stance_category is not None:
            raise ValueError("Non-CLASSIFIED records must not carry a stance category.")

    @property
    def stance_position(self) -> float | None:
        return None if self.stance_category is None else (self.stance_category - 1) / 4
