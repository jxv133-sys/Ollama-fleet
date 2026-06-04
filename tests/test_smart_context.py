"""Tests for SmartContextBuilder."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.smart_context import SmartContextBuilder
from ollama_fleet.orchestrator.context_builder import ContextBuilder
from ollama_fleet.orchestrator.model_router import ModelRouter
from ollama_fleet.memory.project_memory import ProjectMemoryManager


@pytest.fixture
async def test_db():
    """Create a test database."""
    db = Database(Path(":memory:"))
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
async def smart_context(test_db):
    """Create a SmartContextBuilder for testing."""
    memory = ProjectMemoryManager(test_db)
    context_builder = ContextBuilder(memory)
    model_router = ModelRouter(settings=None)
    
    return SmartContextBuilder(memory, context_builder, model_router)


@pytest.mark.asyncio
async def test_build_for_planning(smart_context):
    """Test building context for planning."""
    context = await smart_context.build_for_planning(
        job_id="test_job",
        goal="Build a web API",
        existing_context="",
    )
    
    assert context["goal"] == "Build a web API"
    assert "focus" in context
    assert "break goal into files" in context["focus"]


@pytest.mark.asyncio
async def test_build_for_code_generation(smart_context):
    """Test building context for code generation."""
    context = await smart_context.build_for_code_generation(
        job_id="test_job",
        target_file="main.py",
        target_purpose="Main entry point",
        requirements="Must support Python 3.8+",
    )
    
    assert context["target_file"] == "main.py"
    assert context["target_purpose"] == "Main entry point"
    assert context["requirements"] == "Must support Python 3.8+"
    assert "project_structure" in context
    assert "dependencies" in context
    assert "focus" in context


@pytest.mark.asyncio
async def test_build_for_code_review(smart_context):
    """Test building context for code review."""
    source_code = "def hello():\n    return 'world'"
    
    context = await smart_context.build_for_code_review(
        job_id="test_job",
        file_path="test.py",
        source_code=source_code,
        requirements="Must return string",
    )
    
    assert context["file_path"] == "test.py"
    assert context["source_code"] == source_code
    assert context["requirements"] == "Must return string"
    assert "auto_approved" in context


@pytest.mark.asyncio
async def test_auto_approve_no_requirements_or_failures(smart_context):
    """Test that code is auto-approved if no requirements or failures."""
    source_code = "def hello():\n    return 'world'"
    
    context = await smart_context.build_for_code_review(
        job_id="test_job",
        file_path="test.py",
        source_code=source_code,
    )
    
    assert context["auto_approved"] is True


@pytest.mark.asyncio
async def test_no_auto_approve_with_requirements(smart_context):
    """Test that code is not auto-approved if requirements exist."""
    source_code = "def hello():\n    return 42"
    
    context = await smart_context.build_for_code_review(
        job_id="test_job",
        file_path="test.py",
        source_code=source_code,
        requirements="Must return string not int",
    )
    
    assert context["auto_approved"] is False


@pytest.mark.asyncio
async def test_build_for_test_analysis(smart_context):
    """Test building context for test analysis."""
    test_output = "FAILED test_main.py::test_hello - AssertionError"
    
    context = await smart_context.build_for_test_analysis(
        job_id="test_job",
        test_output=test_output,
    )
    
    assert context["test_output"] == test_output
    assert "focus" in context
    assert "identify failing tests" in context["focus"]


@pytest.mark.asyncio
async def test_build_for_file_fix(smart_context):
    """Test building context for file fix."""
    context = await smart_context.build_for_file_fix(
        job_id="test_job",
        file_path="main.py",
        error_message="Syntax error: invalid syntax",
        original_purpose="Main entry point",
    )
    
    assert context["target_file"] == "main.py"
    assert context["target_purpose"] == "Main entry point"
    assert "error_context" in context
    assert "Syntax error" in context["error_context"]
    assert "regenerate_instructions" in context


@pytest.mark.asyncio
async def test_validate_code_empty(smart_context):
    """Test validation of empty code."""
    is_valid, error = await smart_context.validate_code_before_review(
        "test.py",
        "",
    )
    
    assert not is_valid
    assert len(error) > 0


@pytest.mark.asyncio
async def test_validate_code_valid_python(smart_context):
    """Test validation of valid Python code."""
    code = "def hello():\n    return 'world'\n"
    is_valid, error = await smart_context.validate_code_before_review(
        "test.py",
        code,
    )
    
    assert is_valid
    assert error == ""


@pytest.mark.asyncio
async def test_validate_code_invalid_python(smart_context):
    """Test validation of invalid Python code."""
    code = "def hello()\n    return 'world'"  # Missing colon
    is_valid, error = await smart_context.validate_code_before_review(
        "test.py",
        code,
    )
    
    assert not is_valid


def test_describe_project_structure(smart_context):
    """Test project structure description."""
    # Mock file metadata
    mock_file1 = MagicMock()
    mock_file1.file_path = "main.py"
    mock_file1.exports = ["run", "main"]
    mock_file1.classes = ["App"]
    
    mock_file2 = MagicMock()
    mock_file2.file_path = "utils.py"
    mock_file2.exports = ["helper"]
    mock_file2.classes = []
    
    all_files = [mock_file1, mock_file2]
    description = smart_context._describe_project_structure(all_files, "main.py")
    
    assert "main.py" in description
    assert "GENERATING" in description
    assert "utils.py" in description
    assert "run" in description
    assert "helper" in description
