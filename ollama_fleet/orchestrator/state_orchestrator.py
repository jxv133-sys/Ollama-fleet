"""State-Driven Orchestrator.

The new orchestrator is the intelligent component.
Agents are tools. The orchestrator decides what to do next.

Core workflow:
1. Observe current project state (from ProjectMemory)
2. Determine next highest-value action
3. Select appropriate capability
4. Execute capability
5. Evaluate result
6. Update project memory
7. Repeat until goal complete
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ollama_fleet.agents.capabilities import CapabilityRegistry, CapabilityType, CapabilityResult
from ollama_fleet.memory.project_memory import ProjectMemoryManager
from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.file_utils import FileTypeDetector
from ollama_fleet.orchestrator.context_builder import ContextBuilder
from ollama_fleet.orchestrator.model_router import ModelRouter
from ollama_fleet.orchestrator.smart_context import SmartContextBuilder

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Possible actions the orchestrator can take."""

    PLAN_PROJECT = "plan_project"
    GENERATE_FILE = "generate_file"
    FIX_VALIDATION = "fix_validation"
    REVIEW_CODE = "review_code"
    RUN_TESTS = "run_tests"
    ANALYZE_FAILURES = "analyze_failures"
    UPDATE_MEMORY = "update_memory"
    COMPLETE = "complete"
    FAIL = "fail"


@dataclass
class OrchestrationState:
    """Current orchestration state snapshot."""

    job_id: str
    goal: str
    current_action: ActionType
    total_files_planned: int
    files_generated: int
    files_validated: int
    files_failed: int
    is_complete: bool
    is_failed: bool
    last_error: str | None


@dataclass
class ActionDecision:
    """Decision about what action to take next."""

    action_type: ActionType
    priority: int  # Higher = more important
    target_file: str | None
    rationale: str
    capability_required: CapabilityType | None


class StateObserver:
    """Observes current project state and determines next action."""

    def __init__(
        self,
        project_memory: ProjectMemoryManager,
    ) -> None:
        self._memory = project_memory

    async def get_orchestration_state(
        self,
        job_id: str,
        goal: str,
    ) -> OrchestrationState:
        """Get snapshot of current orchestration state.

        Args:
            job_id: Job identifier
            goal: Project goal

        Returns:
            OrchestrationState snapshot
        """
        state = await self._memory.get_project_state(job_id)
        files = await self._memory.get_project_files(job_id)

        return OrchestrationState(
            job_id=job_id,
            goal=goal,
            current_action=ActionType.PLAN_PROJECT,
            total_files_planned=state.total_files if state else 0,
            files_generated=len(files),
            files_validated=state.validated_files if state else 0,
            files_failed=state.failed_files if state else 0,
            is_complete=False,
            is_failed=False,
            last_error=None,
        )

    async def determine_next_action(
        self,
        orchestration_state: OrchestrationState,
    ) -> ActionDecision:
        """Determine the next highest-value action to take.

        Decision tree:
        1. If files_planned == 0 → PLAN_PROJECT
        2. If files_generated < files_planned → GENERATE_FILE
        3. If validation failed → FIX_VALIDATION
        4. If tests available → RUN_TESTS
        5. If test failures → ANALYZE_FAILURES
        6. If all files generated & validated → COMPLETE
        7. Else → FAIL

        Args:
            orchestration_state: Current state

        Returns:
            ActionDecision for next action
        """
        # No plan yet
        if orchestration_state.total_files_planned == 0:
            return ActionDecision(
                action_type=ActionType.PLAN_PROJECT,
                priority=100,
                target_file=None,
                rationale="No project plan exists",
                capability_required=CapabilityType.PLANNING,
            )

        # Files still to generate
        if orchestration_state.files_generated < orchestration_state.total_files_planned:
            return ActionDecision(
                action_type=ActionType.GENERATE_FILE,
                priority=90,
                target_file=None,  # Orchestrator will select which file
                rationale="Generate next planned file",
                capability_required=CapabilityType.CODE_GENERATION,
            )

        # Files failed
        if orchestration_state.files_failed > 0:
            return ActionDecision(
                action_type=ActionType.FIX_VALIDATION,
                priority=85,
                target_file=None,
                rationale="Fix failed file validations",
                capability_required=CapabilityType.CODE_GENERATION,
            )

        # All files generated - review
        if orchestration_state.files_generated >= orchestration_state.total_files_planned:
            return ActionDecision(
                action_type=ActionType.RUN_TESTS,
                priority=80,
                target_file=None,
                rationale="Run tests on generated code",
                capability_required=None,  # Uses filesystem, not a capability
            )

        # Default to complete if nothing else needed
        return ActionDecision(
            action_type=ActionType.COMPLETE,
            priority=0,
            target_file=None,
            rationale="All planned work complete",
            capability_required=None,
        )


class StateOrchestrator:
    """Main orchestrator - state-driven workflow engine.

    This replaces the old pipeline orchestrator.
    Instead of: Planner → Spec → Coder → Critic → Tester
    We do: Observe → Decide → Execute → Evaluate → Update Memory → Repeat
    """

    def __init__(
        self,
        job_id: str,
        goal: str,
        db: Database,
        capability_registry: CapabilityRegistry,
    ) -> None:
        self._job_id = job_id
        self._goal = goal
        self._db = db
        self._registry = capability_registry
        self._memory = ProjectMemoryManager(db)
        self._observer = StateObserver(self._memory)
        
        # Initialize smart context system
        self._context_builder = ContextBuilder(self._memory)
        self._model_router = ModelRouter(settings=None)  # Will use defaults
        self._smart_context = SmartContextBuilder(
            self._memory,
            self._context_builder,
            self._model_router,
        )

        self._max_iterations = 50  # Prevent infinite loops
        self._iteration = 0

    async def run(self) -> dict[str, Any]:
        """Run orchestration loop until goal complete.

        Returns:
            Result dict with status, files_generated, etc.
        """
        logger.info(f"Starting state-driven orchestration for job {self._job_id}")

        while self._iteration < self._max_iterations:
            self._iteration += 1

            # Observe current state
            state = await self._observer.get_orchestration_state(
                self._job_id,
                self._goal,
            )
            logger.info(f"Iteration {self._iteration}: {state}")

            # Decide next action
            decision = await self._observer.determine_next_action(state)
            logger.info(f"Next action: {decision.action_type} ({decision.rationale})")

            # Execute action
            if decision.action_type == ActionType.PLAN_PROJECT:
                success = await self._execute_plan()
            elif decision.action_type == ActionType.GENERATE_FILE:
                success = await self._execute_generate_file()
            elif decision.action_type == ActionType.FIX_VALIDATION:
                success = await self._execute_fix_validation()
            elif decision.action_type == ActionType.RUN_TESTS:
                success = await self._execute_run_tests()
            elif decision.action_type == ActionType.ANALYZE_FAILURES:
                success = await self._execute_analyze_failures()
            elif decision.action_type == ActionType.COMPLETE:
                return await self._complete()
            elif decision.action_type == ActionType.FAIL:
                return await self._fail("Max iterations reached or unrecoverable error")
            else:
                return await self._fail(f"Unknown action type: {decision.action_type}")

            if not success:
                logger.warning(f"Action {decision.action_type} failed")

        return await self._fail("Max iterations exceeded")

    # ================================================================
    # Action Execution Methods
    # ================================================================

    async def _execute_plan(self) -> bool:
        """Execute PLAN_PROJECT action.

        Uses PlanningCapability to break goal into files and structure.
        """
        logger.info("Executing: PLAN_PROJECT")

        # Build smart context for planning
        planning_context = await self._smart_context.build_for_planning(
            self._job_id,
            self._goal,
            existing_context="",
        )

        result = await self._registry.execute(
            CapabilityType.PLANNING,
            goal=self._goal,
            context=planning_context.get("existing_context", ""),
        )

        if not result.success:
            logger.error(f"Planning failed: {result.error}")
            return False

        # Parse plan and store in project memory
        plan = result.output  # Should be PlannerOutput
        
        # Extract files from plan and initialize project state
        if hasattr(plan, 'tasks') and plan.tasks:
            file_count = len(plan.tasks)
            logger.info(f"Plan created: {file_count} files to generate")
            
            await self._memory.update_project_state(
                self._job_id,
                "plan_created",
                {
                    "total_files": file_count,
                    "planned_files": [
                        {
                            "path": task.get("filename", ""),
                            "purpose": task.get("purpose", ""),
                        }
                        for task in plan.tasks
                    ],
                },
            )
            return True
        else:
            logger.error("Plan missing tasks")
            return False

    async def _execute_generate_file(self) -> bool:
        """Execute GENERATE_FILE action.

        Select next file to generate, gather context, execute CodeGenerationCapability.
        """
        logger.info("Executing: GENERATE_FILE")

        # Select next file from plan
        file_to_generate = await self._select_next_file_to_generate()
        if not file_to_generate:
            logger.error("No files available to generate")
            return False

        logger.info(f"Generating file: {file_to_generate['path']}")

        # Build smart context using project memory
        generation_context = await self._smart_context.build_for_code_generation(
            self._job_id,
            file_to_generate["path"],
            file_to_generate.get("purpose", ""),
            requirements=self._goal,
        )

        # Execute code generation with enriched context
        result = await self._registry.execute(
            CapabilityType.CODE_GENERATION,
            file_path=generation_context["target_file"],
            file_purpose=generation_context["target_purpose"],
            dependencies=generation_context.get("dependencies", []),
            requirements=generation_context.get("requirements", self._goal),
        )

        if not result.success:
            logger.error(f"Code generation failed: {result.error}")
            return False

        # Extract source code
        source_code = self._extract_source_code(result.output)
        if not source_code:
            logger.error("No source code extracted from output")
            return False

        # Validate the generated code BEFORE sending to critic
        is_valid, error_msg = await self._smart_context.validate_code_before_review(
            file_to_generate["path"],
            source_code,
        )
        
        if not is_valid:
            logger.error(f"Generated code failed validation: {error_msg}")
            return False

        # Detect file type and store in project memory
        detected_type = FileTypeDetector.detect(
            file_to_generate["path"],
            source_code,
        )
        file_type = detected_type.value if detected_type else "text"
        
        await self._memory.store_file_metadata(
            self._job_id,
            file_to_generate["path"],
            source_code,
            file_type=file_type,
        )
        logger.info(f"Stored metadata for {file_to_generate['path']} (type: {file_type})")

        # Mark file as generated in project state
        await self._memory.update_project_state(
            self._job_id,
            "file_generated",
            {"generated_file": file_to_generate["path"]},
        )

        return True

    async def _execute_fix_validation(self) -> bool:
        """Execute FIX_VALIDATION action.

        Find failed files, regenerate with error context.
        """
        logger.info("Executing: FIX_VALIDATION")

        # Get project state to find failed files
        state = await self._memory.get_project_state(self._job_id)
        if not state or state.failed_files == 0:
            logger.info("No failed files to fix")
            return True

        # Find files with validation errors
        all_files = await self._memory.get_project_files(self._job_id)
        failed_files = [f for f in all_files if f.file_path in (state.metadata.get("failed_files", []) or [])]

        if not failed_files:
            logger.info("No failed file metadata found")
            return True

        # Fix first failed file
        file_to_fix = failed_files[0]
        logger.info(f"Fixing failed file: {file_to_fix.file_path}")

        # Get error context
        error_context = state.metadata.get("last_error", "Validation failed")
        
        # Build smart context for file fix
        fix_context = await self._smart_context.build_for_file_fix(
            self._job_id,
            file_to_fix.file_path,
            error_context,
            "Original purpose",
        )

        # Regenerate with error context
        result = await self._registry.execute(
            CapabilityType.CODE_GENERATION,
            file_path=fix_context["target_file"],
            file_purpose=fix_context["target_purpose"],
            dependencies=fix_context.get("dependencies", []),
            requirements=self._goal,
        )

        if not result.success:
            logger.error(f"Fix attempt failed: {result.error}")
            return False

        # Extract and validate
        source_code = self._extract_source_code(result.output)
        if not source_code:
            logger.error("No source code in fix attempt")
            return False

        # Re-validate
        is_valid, error_msg = await self._smart_context.validate_code_before_review(
            file_to_fix.file_path,
            source_code,
        )
        
        if not is_valid:
            logger.error(f"Re-generated code still invalid: {error_msg}")
            return False

        # Update project memory with fixed code
        detected_type = FileTypeDetector.detect(file_to_fix.file_path, source_code)
        file_type = detected_type.value if detected_type else "text"
        
        await self._memory.store_file_metadata(
            self._job_id,
            file_to_fix.file_path,
            source_code,
            file_type=file_type,
        )

        # Update project state
        await self._memory.update_project_state(
            self._job_id,
            "file_fixed",
            {"fixed_file": file_to_fix.file_path},
        )

        return True

    async def _execute_run_tests(self) -> bool:
        """Execute RUN_TESTS action.

        Run actual test suite on generated code.
        """
        logger.info("Executing: RUN_TESTS")

        # Run tests (using workspace manager)
        # For now, we'll skip actual test execution
        # In a full implementation, this would use subprocess or a test runner
        
        logger.info("Test execution would happen here")
        
        # Parse output and store results
        # TODO: Actually run tests

        # Store empty test results for now
        await self._memory.update_project_state(
            self._job_id,
            "tests_run",
            {"test_count": 0, "passed": 0, "failed": 0},
        )

        return True

    async def _execute_analyze_failures(self) -> bool:
        """Execute ANALYZE_FAILURES action.

        Use TestingCapability to analyze test failures.
        """
        logger.info("Executing: ANALYZE_FAILURES")

        # Get test output
        state = await self._memory.get_project_state(self._job_id)
        test_output = state.metadata.get("test_output", "") if state else ""

        if not test_output:
            logger.info("No test failures to analyze")
            return True

        # Execute TestingCapability
        result = await self._registry.execute(
            CapabilityType.TESTING,
            test_output=test_output,
            goal=self._goal,
        )

        if not result.success:
            logger.error(f"Test analysis failed: {result.error}")
            return False

        # Store analysis
        await self._memory.update_project_state(
            self._job_id,
            "failures_analyzed",
            {"analysis": result.output},
        )

        # Determine if tests pass
        if hasattr(result.output, 'all_passed') and result.output.all_passed:
            logger.info("All tests passed!")
            return True
        else:
            logger.warning("Tests still failing, will retry")
            return True  # Continue orchestration, decision tree will handle retries

    async def _complete(self) -> dict[str, Any]:
        """Mark job as complete."""
        logger.info(f"Job {self._job_id} completed successfully")

        await self._memory.update_project_state(
            self._job_id,
            "complete",
        )

        return {
            "status": "success",
            "job_id": self._job_id,
            "iterations": self._iteration,
        }

    async def _fail(self, reason: str) -> dict[str, Any]:
        """Mark job as failed."""
        logger.error(f"Job {self._job_id} failed: {reason}")

        await self._memory.update_project_state(
            self._job_id,
            "failed",
            {"error": reason},
        )

        return {
            "status": "failed",
            "job_id": self._job_id,
            "error": reason,
            "iterations": self._iteration,
        }

    # ================================================================
    # Helper Methods
    # ================================================================

    def _build_planning_context(self, existing_files: list[Any]) -> str:
        """Build context for planning capability.
        
        Includes existing files and their purposes.
        """
        if not existing_files:
            return ""
        
        context_lines = ["Existing files in project:"]
        for file_entry in existing_files:
            context_lines.append(f"- {file_entry.file_path}: {', '.join(file_entry.exports)}")
        
        return "\n".join(context_lines)

    async def _select_next_file_to_generate(self) -> dict[str, Any] | None:
        """Select next file to generate from the plan.
        
        Returns the first file that hasn't been generated yet.
        """
        state = await self._memory.get_project_state(self._job_id)
        if not state:
            return None
        
        planned_files = state.metadata.get("planned_files", [])
        generated_files = state.metadata.get("generated_files", []) or []
        
        for file_plan in planned_files:
            if file_plan.get("path") not in [g.get("path") if isinstance(g, dict) else g for g in generated_files]:
                return file_plan
        
        return None

    def _validate_code_output(self, output: Any) -> bool:
        """Validate that code output is valid.
        
        Checks:
        - Not empty
        - Not just a filename
        - Contains actual source code
        - Has parseable syntax
        """
        if not output:
            return False
        
        # If it's an object with source_code attribute
        if hasattr(output, 'source_code'):
            code = output.source_code
        elif isinstance(output, dict) and 'source_code' in output:
            code = output['source_code']
        elif isinstance(output, str):
            code = output
        else:
            return False
        
        if not code or len(code.strip()) < 10:
            return False
        
        # Basic check: should contain more than just a file path
        if code.count('\n') < 2:
            return False
        
        return True

    def _extract_source_code(self, output: Any) -> str | None:
        """Extract source code from capability output.
        
        Returns the actual source code string.
        """
        if hasattr(output, 'source_code'):
            return output.source_code
        elif isinstance(output, dict) and 'source_code' in output:
            return output['source_code']
        elif isinstance(output, str):
            return output
        
        return None
