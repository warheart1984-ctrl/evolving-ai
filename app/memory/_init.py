from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class LessonStatus(Enum):
    """Lifecycle states for lessons."""
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    QUARANTINED = "quarantined"


class Lesson(BaseModel):
    """A lesson learned from operator execution or evaluation."""
    id: str
    claim: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    scope: str  # e.g., "database_tasks", "api_calls", "math_reasoning"
    created_by: str  # e.g., "steward:v17", "operator:v15"
    status: LessonStatus = LessonStatus.CANDIDATE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Lineage
    parent_lesson_id: Optional[str] = None
    triggering_task_id: Optional[str] = None
    
    # Metadata
    times_validated: int = Field(default=0)
    times_quarantined: int = Field(default=0)
    validated_by: Optional[str] = None
    validated_at: Optional[datetime] = None
    quarantine_reason: Optional[str] = None
    quarantined_by: Optional[str] = None
    quarantined_at: Optional[datetime] = None


class MemoryStore:
    """Governed memory with lesson lifecycle management."""
    
    def __init__(self, persistence=None):
        self._lessons: Dict[str, Lesson] = {}
        self._next_id = 1
        self._persist = persistence
        if persistence:
            self._load_from_persistence()

    def _load_from_persistence(self):
        rows = self._persist.load_all("lesson")
        for lesson_id, data in rows.items():
            try:
                self._lessons[lesson_id] = Lesson(**data)
                num = int(lesson_id.split("_")[-1]) + 1
                if num > self._next_id:
                    self._next_id = num
            except Exception:
                pass

    def _persist_lesson(self, lesson: Lesson):
        if self._persist:
            self._persist.save("lesson", lesson.id, lesson.model_dump(mode="json"))
    
    def create_lesson(
        self,
        claim: str,
        scope: str,
        created_by: str,
        triggering_task_id: Optional[str] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Lesson:
        """Create a new lesson in candidate state."""
        lesson_id = f"lesson_{self._next_id:04d}"
        self._next_id += 1
        
        lesson = Lesson(
            id=lesson_id,
            claim=claim,
            evidence=evidence or [],
            scope=scope,
            created_by=created_by,
            status=LessonStatus.CANDIDATE,
            triggering_task_id=triggering_task_id,
        )
        
        self._lessons[lesson_id] = lesson
        self._persist_lesson(lesson)
        return lesson
    
    def validate_lesson(self, lesson_id: str, validator_id: str) -> bool:
        """Move a lesson from candidate to validated state."""
        lesson = self._lessons.get(lesson_id)
        if not lesson or lesson.status != LessonStatus.CANDIDATE:
            return False
        
        lesson.status = LessonStatus.VALIDATED
        lesson.times_validated += 1
        lesson.validated_by = validator_id
        lesson.validated_at = datetime.utcnow()
        self._persist_lesson(lesson)
        return True
    
    def activate_lesson(self, lesson_id: str) -> bool:
        """Move a lesson from validated to active state."""
        lesson = self._lessons.get(lesson_id)
        if not lesson or lesson.status != LessonStatus.VALIDATED:
            return False
        
        lesson.status = LessonStatus.ACTIVE
        self._persist_lesson(lesson)
        return True
    
    def quarantine_lesson(self, lesson_id: str, reason: str, quarantiner_id: str) -> bool:
        """Move a lesson to quarantined state."""
        lesson = self._lessons.get(lesson_id)
        if not lesson:
            return False
        
        lesson.status = LessonStatus.QUARANTINED
        lesson.quarantine_reason = reason
        lesson.quarantined_by = quarantiner_id
        lesson.quarantined_at = datetime.utcnow()
        lesson.times_quarantined += 1
        self._persist_lesson(lesson)
        return True
    
    def deactivate_lesson(self, lesson_id: str) -> bool:
        """Move an active lesson to deprecated state."""
        lesson = self._lessons.get(lesson_id)
        if not lesson or lesson.status != LessonStatus.ACTIVE:
            return False
        
        lesson.status = LessonStatus.DEPRECATED
        self._persist_lesson(lesson)
        return True
    
    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """Get a lesson by ID."""
        return self._lessons.get(lesson_id)
    
    def get_lessons_by_status(self, status: LessonStatus) -> List[Lesson]:
        """Get all lessons with a specific status."""
        return [l for l in self._lessons.values() if l.status == status]
    
    def get_lessons_by_scope(self, scope: str) -> List[Lesson]:
        """Get all lessons for a specific scope."""
        return [l for l in self._lessons.values() if l.scope == scope]
    
    def get_active_lessons(self) -> List[Lesson]:
        """Get all active lessons."""
        return self.get_lessons_by_status(LessonStatus.ACTIVE)
    
    def get_validated_lessons(self) -> List[Lesson]:
        """Get all validated lessons."""
        return self.get_lessons_by_status(LessonStatus.VALIDATED)