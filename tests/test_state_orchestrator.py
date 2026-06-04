"""Tests for the new state-driven orchestrator."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.state_orchestrator import (
    StateOrchestrator,
    StateObserver,
    ActionType,
)
from ollama_fleet.agents.capabilities import CapabilityRegistry
from ollama_fleet.memory.project_memory import ProjectMemoryManager


@pytest.fixture
async def test_db():
    """Create a test database."""
    db = Database(Path(":memory:"))
    await db.connect()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_state_observer_initial_state(test_db):
    """Test StateObserver can read initial project state."""
    memory = ProjectMemoryManager(test_db)
    observer = StateObserver(memory)
    
    # Get initial state for a new job
    state = await observer.get_orchestration_state(
        job_id="test_job",
        goal="Build test project",
    )
    
    assert state.job_id == "test_job"
    assert state.goal == "Build test project"
    assert state.total_files_planned == 0
    assert state.files_generated == 0


@pytest.mark.asyncio
async def test_state_observer_decision_with_no_plan(test_db):
    """Test StateObserver decides to plan when no plan exists."""
    memory = ProjectMemoryManager(test_db)
    observer = StateObserver(memory)
    
    state = await observer.get_orchestration_state(
        job_id="test_job",
        goal="Build test project",
    )
    
    decision = await observer.determine_next_action(state)
    
    assert decision.action_type == ActionType.PLAN_PROJECT
    assert decision.capability_required is not None


@pytest.mark.asyncio
async def test_orchestrator_instantiation(test_db):
    """Test StateOrchestrator can be instantiated."""
    # Create mock capability registry
    mock_registry = MagicMock(spec=CapabilityRegistry)
    
    orchestrator = StateOrchestrator(
        job_id="test_job",
        goal="Build test project",
        db=test_db,
        capability_registry=mock_registry,
    )
    
    assert orchestrator._job_id == "test_job"
    assert orchestrator._goal == "Build test project"


@pytest.mark.asyncio
async def test_validation_code_output(test_db):
    """Test code validation logic."""
    mock_registry = MagicMock(spec=CapabilityRegistry)
    orchestrator = StateOrchestrator(
        job_id="test_job",
        goal="Build test project",
        db=test_db,
        capability_registry=mock_registry,
    )
    
    # Test empty code
    assert not orchestrator._validate_code_output("")
    
    # Test code-like string (needs multiple lines)
    multi_line_code = "def hello():\n    return 'world'\n\nprint(hello())"
    assert orchestrator._validate_code_output(multi_line_code)
    
    # Test just a filename
    assert not orchestrator._validate_code_output("main.py")


@pytest.mark.asyncio
async def test_extract_source_code(test_db):
    """Test source code extraction from various output formats."""
    mock_registry = MagicMock(spec=CapabilityRegistry)
    orchestrator = StateOrchestrator(
        job_id="test_job",
        goal="Build test project",
        db=test_db,
        capability_registry=mock_registry,
    )
    
    # Test string output
    code_str = "def hello():\n    pass"
    assert orchestrator._extract_source_code(code_str) == code_str
    
    # Test dict output
    code_dict = {"source_code": code_str}
    assert orchestrator._extract_source_code(code_dict) == code_str
    
    # Test object with source_code attribute
    mock_output = MagicMock()
    mock_output.source_code = code_str
    assert orchestrator._extract_source_code(mock_output) == code_str


@pytest.mark.asyncio
async def test_helper_methods(test_db):
    """Test helper methods."""
    mock_registry = MagicMock(spec=CapabilityRegistry)
    orchestrator = StateOrchestrator(
        job_id="test_job",
        goal="Build test project",
        db=test_db,
        capability_registry=mock_registry,
    )
    
    # Test planning context building
    context = orchestrator._build_planning_context([])
    assert context == ""
    
    # Test next file selection (no files planned)
    next_file = await orchestrator._select_next_file_to_generate()
    assert next_file is None


def test_action_types():
    """Test ActionType enum has all expected values."""
    assert hasattr(ActionType, "PLAN_PROJECT")
    assert hasattr(ActionType, "GENERATE_FILE")
    assert hasattr(ActionType, "FIX_VALIDATION")
    assert hasattr(ActionType, "RUN_TESTS")
    assert hasattr(ActionType, "ANALYZE_FAILURES")
    assert hasattr(ActionType, "COMPLETE")
    assert hasattr(ActionType, "FAIL")
