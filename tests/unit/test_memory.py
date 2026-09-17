"""Tests for memory module - lesson lifecycle management."""
import pytest
from datetime import datetime
from app.memory._init import MemoryStore, Lesson, LessonStatus


class TestMemoryStore:
    """Test memory lesson lifecycle."""
    
    def test_create_lesson_candidate(self):
        """Test creating a lesson starts in candidate state."""
        store = MemoryStore()
        lesson = store.create_lesson(
            claim="When X happens, strategy Y improves result Z",
            scope="database_tasks",
            created_by="steward:v17",
            triggering_task_id="task-001",
        )
        
        assert lesson.id.startswith("lesson_")
        assert lesson.status == LessonStatus.CANDIDATE
        assert lesson.claim == "When X happens, strategy Y improves result Z"
        assert lesson.scope == "database_tasks"
        assert lesson.created_by == "steward:v17"
    
    def test_lesson_lifecycle_candidate_to_validated(self):
        """Test moving lesson from candidate to validated."""
        store = MemoryStore()
        lesson = store.create_lesson(
            claim="Test claim",
            scope="test_scope",
            created_by="steward:v17",
        )
        
        # Validate
        success = store.validate_lesson(lesson.id, "validator:v18")
        assert success == True
        
        # Check state
        updated = store.get_lesson(lesson.id)
        assert updated.status == LessonStatus.VALIDATED
        assert updated.times_validated == 1
    
    def test_lesson_lifecycle_validated_to_active(self):
        """Test moving lesson from validated to active."""
        store = MemoryStore()
        lesson = store.create_lesson(
            claim="Test claim",
            scope="test_scope",
            created_by="steward:v17",
        )
        
        # Validate first
        store.validate_lesson(lesson.id, "validator:v18")
        
        # Activate
        success = store.activate_lesson(lesson.id)
        assert success == True
        
        # Check state
        updated = store.get_lesson(lesson.id)
        assert updated.status == LessonStatus.ACTIVE
    
    def test_lesson_lifecycle_active_to_deprecated(self):
        """Test moving lesson from active to deprecated."""
        store = MemoryStore()
        lesson = store.create_lesson(
            claim="Test claim",
            scope="test_scope",
            created_by="steward:v17",
        )
        
        # Full lifecycle
        store.validate_lesson(lesson.id, "validator:v18")
        store.activate_lesson(lesson.id)
        
        # Deprecate
        success = store.deactivate_lesson(lesson.id)
        assert success == True
        
        # Check state
        updated = store.get_lesson(lesson.id)
        assert updated.status == LessonStatus.DEPRECATED
    
    def test_lesson_lifecycle_quarantine(self):
        """Test quarantining a lesson."""
        store = MemoryStore()
        lesson = store.create_lesson(
            claim="Test claim",
            scope="test_scope",
            created_by="steward:v17",
        )
        
        # Quarantine
        success = store.quarantine_lesson(lesson.id, " insufficient evidence", "validator:v18")
        assert success == True
        
        # Check state
        updated = store.get_lesson(lesson.id)
        assert updated.status == LessonStatus.QUARANTINED
        assert updated.quarantine_reason == " insufficient evidence"
        assert updated.times_quarantined == 1
    
    def test_get_lessons_by_status(self):
        """Test filtering lessons by status."""
        store = MemoryStore()
        
        # Create multiple lessons
        l1 = store.create_lesson("Claim 1", "scope1", "steward:v17")
        l2 = store.create_lesson("Claim 2", "scope2", "steward:v17")
        l3 = store.create_lesson("Claim 3", "scope1", "steward:v17")
        
        # Validate l1
        store.validate_lesson(l1.id, "validator:v18")
        
        # Get validated lessons
        validated = store.get_lessons_by_status(LessonStatus.VALIDATED)
        assert len(validated) == 1
        assert validated[0].id == l1.id
    
    def test_get_lessons_by_scope(self):
        """Test filtering lessons by scope."""
        store = MemoryStore()
        
        # Create lessons in different scopes
        l1 = store.create_lesson("DB claim", "database_tasks", "steward:v17")
        l2 = store.create_lesson("API claim", "api_calls", "steward:v17")
        l3 = store.create_lesson("Math claim", "math_reasoning", "steward:v17")
        
        # Get database tasks lessons
        db_lessons = store.get_lessons_by_scope("database_tasks")
        assert len(db_lessons) == 1
        assert db_lessons[0].claim == "DB claim"
    
    def test_get_active_lessons(self):
        """Test getting all active lessons."""
        store = MemoryStore()
        
        # Create and activate some lessons
        l1 = store.create_lesson("Active 1", "scope1", "steward:v17")
        store.validate_lesson(l1.id, "validator:v18")
        store.activate_lesson(l1.id)
        
        l2 = store.create_lesson("Candidate 2", "scope2", "steward:v17")
        # l2 stays as candidate
        
        active = store.get_active_lessons()
        assert len(active) == 1
        assert active[0].id == l1.id