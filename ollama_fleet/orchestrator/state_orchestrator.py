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

        result = await self._registry.execute(
            CapabilityType.PLANNING,
            goal=self._goal,
            context="",
        )

        if not result.success:
            logger.error(f"Planning failed: {result.error}")
            return False

        # Parse plan and store in project memory
        plan = result.output  # Should be PlannerOutput
        # TODO: Extract files from plan and initialize project state

        await self._memory.update_project_state(
            self._job_id,
            "plan_created",
            {"total_files": len(plan.tasks)},
        )

        return True

    async def _execute_generate_file(self) -> bool:
        """Execute GENERATE_FILE action.

        Select next file to generate, gather context, execute CodeGenerationCapability.
        """
        logger.info("Executing: GENERATE_FILE")

        # TODO: Select next file from plan
        # TODO: Gather context (dependencies, project memory)
        # TODO: Execute code generation
        # TODO: Store in project memory
        # TODO: Mark file as generated

        return True

    async def _execute_fix_validation(self) -> bool:
        """Execute FIX_VALIDATION action.

        Find failed files, regenerate with error context.
        """
        logger.info("Executing: FIX_VALIDATION")

        # TODO: Find files with validation errors
        # TODO: Regenerate with error context
        # TODO: Re-validate

        return True

    async def _execute_run_tests(self) -> bool:
        """Execute RUN_TESTS action.

        Run actual test suite on generated code.
        """
        logger.info("Executing: RUN_TESTS")

        # TODO: Run tests (using workspace manager)
        # TODO: Parse output
        # TODO: Store results

        return True

    async def _execute_analyze_failures(self) -> bool:
        """Execute ANALYZE_FAILURES action.

        Use TestingCapability to analyze test failures.
        """
        logger.info("Executing: ANALYZE_FAILURES")

        # TODO: Get test output
        # TODO: Execute TestingCapability
        # TODO: Store analysis
        # TODO: Determine if tests pass

        return True

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
