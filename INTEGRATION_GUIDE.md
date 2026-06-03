# Integration Guide: New Orchestrator Architecture

This guide shows how to integrate the new components into the existing Ollama Fleet system.

## Overview

The new architecture has these core components ready:

1. **ProjectMemoryManager** - Persistent project state index
2. **CapabilityRegistry** - Tool/capability management
3. **StateOrchestrator** - Main orchestration loop
4. **ContextBuilder** - Intelligent context for models
5. **ModelRouter** - Model selection per task
6. **ValidationLayer** - Pre-critic code validation

## Integration Steps

### Step 1: Update Main Orchestrator Initialization

**File:** `ollama_fleet/orchestrator/orchestrator.py`

**Current:** Uses Planner → Spec → Coder → Critic pipeline

**Change:** Support both old and new orchestration modes

```python
from ollama_fleet.orchestrator.state_orchestrator import StateOrchestrator
from ollama_fleet.orchestrator.model_router import ModelRouter
from ollama_fleet.orchestrator.context_builder import ContextBuilder
from ollama_fleet.agents.capabilities import CapabilityRegistry
from ollama_fleet.memory.project_memory import ProjectMemoryManager

class Orchestrator:
    def __init__(self, db: Database, settings: FleetSettings):
        # ... existing code ...
        
        # NEW: Initialize components
        self._project_memory = ProjectMemoryManager(db)
        self._capability_registry = CapabilityRegistry(self._executor)
        self._model_router = ModelRouter(settings)
        self._context_builder = ContextBuilder(self._project_memory)
        
        # Feature flag for new orchestrator
        self._use_state_orchestrator = settings.get("use_state_orchestrator", False)

    async def submit_job(self, job_id: str, goal: str, config: JobConfig) -> Job:
        if self._use_state_orchestrator:
            # NEW: Use state-driven orchestrator
            orchestrator = StateOrchestrator(
                job_id=job_id,
                goal=goal,
                db=self._db,
                capability_registry=self._capability_registry,
            )
            result = await orchestrator.run()
            return self._handle_result(result)
        else:
            # OLD: Keep existing pipeline for backward compatibility
            return await self._run_pipeline_orchestrator(job_id, goal, config)
```

---

### Step 2: Connect AgentExecutor to ModelRouter

**File:** `ollama_fleet/agents/executor.py`

**Change:** Use ModelRouter to select model before LLM call

```python
from ollama_fleet.orchestrator.model_router import ModelRouter
from ollama_fleet.agents.capabilities import CapabilityType

class AgentExecutor:
    def __init__(self, client: OllamaClient, model_router: ModelRouter = None):
        self._client = client
        self._model_router = model_router

    async def execute_planner(self, goal: str, context: str = "") -> PlannerOutput:
        # NEW: Select model for planning
        model_config = None
        if self._model_router:
            model_config = self._model_router.get_model_for_capability(
                CapabilityType.PLANNING
            )

        # Use selected model instead of default
        model_name = model_config.model_name if model_config else "llama2"
        
        prompt = build_planner_prompt(goal, context)
        output = await self._client.generate(
            model=model_name,
            prompt=prompt,
            max_tokens=model_config.max_tokens if model_config else 2048,
        )
        return parse_planner_output(output)
```

---

### Step 3: Update CodeGeneration to Use Focused Context

**File:** `ollama_fleet/agents/executor.py` or new method in Orchestrator

**Change:** Use ContextBuilder to prepare context

```python
async def execute_coder_with_context(
    self,
    job_id: str,
    file_path: str,
    file_purpose: str,
    requirements: str = "",
):
    # NEW: Build focused context
    context = await self._context_builder.build_context_for_file_generation(
        job_id=job_id,
        target_file=file_path,
        target_purpose=file_purpose,
        requirements=requirements,
    )
    
    # Select model
    model_config = self._model_router.get_model_for_capability(
        CapabilityType.CODE_GENERATION
    )
    
    # Build prompt using focused context
    prompt = build_coder_prompt(
        target_file=context.target_file,
        purpose=context.target_purpose,
        dependencies=context.relevant_dependencies,
        exports=context.relevant_exports,
        requirements=context.explicit_requirements,
    )
    
    output = await self._client.generate(
        model=model_config.model_name,
        prompt=prompt,
        max_tokens=model_config.max_tokens,
    )
    
    return parse_coder_output(output)
```

---

### Step 4: Add Pre-Critic Validation

**File:** `ollama_fleet/orchestrator/context_builder.py` (already has ValidationLayer)

**Change:** Use ValidationLayer before sending to critic

```python
from ollama_fleet.orchestrator.context_builder import ValidationLayer

# After code generation
is_valid, error_message = ValidationLayer.validate_generated_code(
    source_code=generated_code,
    file_path=file_path
)

if not is_valid:
    logger.warning(f"Validation failed: {error_message}")
    # Reject immediately, request regeneration
    # Do NOT send to critic
    return {"approved": False, "reason": f"Validation: {error_message}"}

# Only proceed to critic if validation passes
critic_input = await context_builder.build_context_for_code_review(...)
```

---

### Step 5: Simplify Critic to Only Explicit Issues

**File:** `ollama_fleet/agents/executor.py` and prompt

**Change:** Rewrite critic prompt to only evaluate concrete issues

Old behavior: "Find any bugs, potential issues, style problems..."
New behavior: "Review ONLY: failed tests, explicit requirements, syntax errors"

```python
CRITIC_PROMPT = """
You are a code reviewer. Your job is to evaluate code ONLY against these criteria:

ONLY evaluate if:
1. There are failing tests (provided below)
2. There are explicit requirements (provided below)
3. There are syntax/parse errors

Do NOT invent hypothetical bugs or style issues.

Code:
{source_code}

Explicit Requirements:
{requirements}

Test Failures (if any):
{test_failures}

If NO requirements and NO test failures, respond with APPROVED.
If code has concrete issues, respond with: REJECTED - [issue list]
"""
```

---

### Step 6: Store Generated File Metadata in ProjectMemory

**File:** `ollama_fleet/orchestrator/orchestrator.py` or `StateOrchestrator`

**Change:** After successful file generation, extract and store metadata

```python
# After file written to workspace
source_code = file_contents  # Get generated code

# Store in project memory
await self._project_memory.store_file_metadata(
    job_id=job_id,
    file_path=file_path,
    source_code=source_code,
    file_type="python",
)

# Extract and store interfaces
interfaces = extract_interfaces(source_code, file_path)
await self._project_memory.store_interfaces(
    job_id=job_id,
    file_path=file_path,
    interfaces=interfaces,
)

# Update project state
await self._project_memory.update_project_state(
    job_id=job_id,
    last_action=f"Generated {file_path}",
)
```

---

### Step 7: Implement StateOrchestrator Action Methods

**File:** `ollama_fleet/orchestrator/state_orchestrator.py`

**Status:** Skeleton created, needs implementation

**Missing methods to implement:**

```python
async def _execute_plan(self) -> bool:
    # Use PlanningCapability to create initial plan
    # Parse plan and store in project memory
    # Initialize project state
    pass

async def _execute_generate_file(self) -> bool:
    # Get next file from project memory plan
    # Gather dependencies
    # Execute CodeGenerationCapability
    # Validate output
    # Store metadata
    # Mark file complete
    pass

async def _execute_fix_validation(self) -> bool:
    # Find files with validation errors
    # Provide error context to CodeGenerationCapability
    # Re-validate
    pass

async def _execute_run_tests(self) -> bool:
    # Execute test command on workspace
    # Parse output
    # Store results
    pass

async def _execute_analyze_failures(self) -> bool:
    # Get test output
    # Use TestingCapability to analyze
    # Determine next action
    pass
```

---

## Configuration

Add to `config.toml`:

```toml
[orchestration]
# Use new state-driven orchestrator (default: false for backward compatibility)
use_state_orchestrator = false

# Model routing
planning_model = "llama2"
coding_model = "llama2"
fast_model = "llama2"

[context]
# Max tokens for focused context (prevents token overflow)
max_context_tokens = 4000
```

---

## Testing Strategy

### Unit Tests

1. **ProjectMemoryManager**
   - Store/retrieve file metadata
   - Dependency resolution
   - Interface extraction

2. **ContextBuilder**
   - Build focused context
   - Validate auto-approval logic
   - Dependency filtering

3. **StateOrchestrator**
   - Decision logic (next action selection)
   - State transitions
   - Iteration limits

4. **ModelRouter**
   - Capability → Model routing
   - Fallback handling
   - Configuration override

### Integration Tests

1. Full workflow with new orchestrator
2. Metadata extraction and storage
3. Context building and validation
4. Model routing in actual execution

---

## Backward Compatibility

The new system is designed to coexist with the old one:

- Feature flag: `use_state_orchestrator` (default: false)
- Old pipeline still works
- Gradual migration path
- Can test new system in parallel

To enable new orchestrator:

```python
# In config or env
OLLAMA_FLEET_USE_STATE_ORCHESTRATOR=true
```

---

## Success Criteria

✅ New system reduces error accumulation
✅ Focused context prevents hallucination
✅ Orchestrator intelligently selects next action
✅ ValidationLayer catches invalid output early
✅ ProjectMemory correctly tracks file interfaces
✅ Critics only evaluate explicit issues
✅ Tests pass with new system

---

## Timeline Estimate

- Integration: 2-3 hours
- Implementation of stub methods: 3-4 hours
- Testing: 2-3 hours
- **Total: 1-2 days for full integration**
