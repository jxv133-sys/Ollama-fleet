# Ollama Fleet Refactoring Progress

Date: June 3, 2026

## Architecture Transformation Summary

Transforming from sequential pipeline to state-driven orchestrator model.

### ✅ Phase 1: Project Memory System (COMPLETE)

**Files Created:**
- `ollama_fleet/db/migrations/004_project_memory.sql` - Project memory schema
- `ollama_fleet/memory/project_memory.py` - ProjectMemoryManager class

**Tables Added:**
- `project_memory` - File metadata (exports, imports, classes, functions, dependencies)
- `project_interfaces` - Extracted interfaces (functions, classes with signatures)
- `project_state` - High-level project state snapshot

**Key Classes:**
- `ProjectMemoryManager` - Full API for storing/querying file metadata
- `FileMetadata` - Structured file information
- `ProjectMemoryEntry` - Stored file metadata with timestamps
- `ProjectInterface` - Function/class interface representation
- `ProjectState` - Project state snapshot

**Features:**
- Extract metadata from generated code (imports, exports, classes, functions)
- Query dependencies for a file
- Find interfaces across project
- Track project state (files generated, validated, failed)
- Persistent storage in SQLite

---

### ✅ Phase 2: Convert Agents to Tools (COMPLETE)

**Files Created:**
- `ollama_fleet/agents/capabilities.py` - Capability base classes and implementations

**Key Classes:**
- `Capability` (ABC) - Base class for all tools
- `CapabilityType` (Enum) - PLANNING, CODE_GENERATION, CODE_REVIEW, TESTING, WORKSPACE_SEARCH, INTERFACE_EXTRACTION
- `CapabilityResult` - Standardized result format
- `PlanningCapability` - Plan project structure
- `CodeGenerationCapability` - Generate/modify files
- `CodeReviewCapability` - Review code for explicit issues (no hallucinations)
- `TestingCapability` - Analyze test results
- `WorkspaceSearchCapability` - Search project files
- `InterfaceExtractionCapability` - Extract APIs from code
- `CapabilityRegistry` - Factory for creating and executing capabilities

**Design:**
- Agents are pure tools with no decision-making
- All take standardized inputs, return CapabilityResult
- Removed decision-making logic (planner, specification agent)
- File Specification Agent removed (reduces hallucination layers)

---

### ✅ Phase 3: State-Driven Orchestrator (COMPLETE)

**Files Created:**
- `ollama_fleet/orchestrator/state_orchestrator.py` - New orchestrator engine

**Key Classes:**
- `StateOrchestrator` - Main orchestration loop
- `StateObserver` - Observes project state and determines next action
- `OrchestrationState` - Current state snapshot
- `ActionDecision` - Decision about what to do next
- `ActionType` (Enum) - PLAN_PROJECT, GENERATE_FILE, FIX_VALIDATION, REVIEW_CODE, RUN_TESTS, ANALYZE_FAILURES, UPDATE_MEMORY, COMPLETE, FAIL

**Workflow:**
1. Observe project state (from ProjectMemory)
2. Determine next highest-value action
3. Select appropriate capability
4. Execute capability
5. Evaluate result
6. Update project memory
7. Repeat until complete

**Decision Tree:**
- No plan? → PLAN_PROJECT
- Files to generate? → GENERATE_FILE
- Files failed validation? → FIX_VALIDATION
- Tests available? → RUN_TESTS
- Test failures? → ANALYZE_FAILURES
- All complete? → COMPLETE

---

### ✅ Phase 4: Intelligent Context Builder (COMPLETE)

**Files Created:**
- `ollama_fleet/orchestrator/context_builder.py` - Focused context generation

**Key Classes:**
- `ContextBuilder` - Build minimal, focused context for each task
- `FocusedContext` - Minimal context for file generation
- `ValidationLayer` - Pre-critic validation

**Features:**
- Build context for code generation (only dependencies, not entire project)
- Build context for code review (only explicit requirements/test failures)
- Build context for test analysis (only test output and source files)
- Auto-approval if no requirements and no test failures
- Reject empty/invalid code before sending to critic

**Principle:**
Do NOT pass entire project to model. Only provide information needed for current task.

---

### ✅ Phase 5: Model Router (COMPLETE)

**Files Created:**
- `ollama_fleet/orchestrator/model_router.py` - Model selection per task type

**Key Classes:**
- `ModelRouter` - Routes capabilities to appropriate models
- `ModelConfig` - Configuration for a specific model
- `ModelType` (Enum) - REASONING, CODING, FAST

**Default Routing:**
- Planning → Reasoning model (e.g., claude-opus)
- Code generation → Coding model (e.g., claude-sonnet)
- Code review → Reasoning model
- Testing → Reasoning model
- Interface extraction → Coding model

**Features:**
- Cost/performance optimization
- Swappable models per capability
- Configurable routing table

---

## Remaining Work

### Phase 6: Integrate New Components
- [ ] Update main orchestrator to use StateOrchestrator
- [ ] Integrate ProjectMemoryManager into orchestrator
- [ ] Wire up CapabilityRegistry with AgentExecutor
- [ ] Connect ModelRouter to AgentExecutor

### Phase 7: Update Agents
- [ ] Simplify Planner (structure only, no implementation details)
- [ ] Simplify Coder (use project memory context)
- [ ] Simplify Critic (only explicit issues, no hypotheticals)
- [ ] Update agents to use focused context from ContextBuilder

### Phase 8: Task Scheduling Updates
- [ ] Update TaskScheduler for new action types
- [ ] Remove dependency tracking (orchestrator handles it)
- [ ] Simplify task model

### Phase 9: Testing & Validation
- [ ] Unit tests for ProjectMemoryManager
- [ ] Unit tests for StateOrchestrator decision logic
- [ ] Integration tests for full workflow
- [ ] Validation that errors don't accumulate

### Phase 10: UI Updates
- [ ] Update web_gui.py for action-driven view
- [ ] Display project memory state
- [ ] Show action decisions in real-time
- [ ] Display focused context being used

---

## Key Design Decisions

### 1. Orchestrator is the Brain
All intelligence in orchestrator. Agents are tools. Only orchestrator makes decisions.

### 2. No Hallucination Cascade
Each agent is pure tool. Previous agent output → facts extracted → stored in memory → used as input for next agent.

### 3. Focused Context Only
Don't dump entire project into prompts. Use ProjectMemory to query only needed information.

### 4. Validation Before Critics
If code is invalid (empty, syntax error, not source code), reject immediately. Don't ask critics to review invalid output.

### 5. Critics Don't Hypothesize
Critics only evaluate:
- Failed tests (concrete)
- Explicit requirements (given)
- Validation failures (objective)

Critics never invent bugs that don't exist.

### 6. State-Driven Workflow
Instead of linear pipeline, orchestrator observes state and decides what's needed next. More flexible, handles partial failures better.

### 7. Model Routing
Different tasks use different models. Optimize cost and quality per capability.

---

## Architecture Diagram

```
User Goal
    ↓
StateOrchestrator (the brain)
    ├─→ StateObserver (observe project state from ProjectMemory)
    ├─→ Decision: "What action is needed next?"
    ├─→ ModelRouter (select model for this capability)
    ├─→ ContextBuilder (build minimal focused context)
    ├─→ CapabilityRegistry.execute (call the tool)
    │   ├─→ PlanningCapability
    │   ├─→ CodeGenerationCapability
    │   ├─→ CodeReviewCapability
    │   ├─→ TestingCapability
    │   └─→ ...
    ├─→ ValidationLayer (reject invalid output)
    ├─→ ProjectMemoryManager (store extracted metadata)
    ├─→ Update project state
    └─→ Repeat until goal complete
```

---

## Files Changed Summary

**New Files (10):**
1. `ollama_fleet/db/migrations/004_project_memory.sql` - DB schema
2. `ollama_fleet/memory/project_memory.py` - Project memory manager
3. `ollama_fleet/agents/capabilities.py` - Capability system
4. `ollama_fleet/orchestrator/state_orchestrator.py` - New orchestrator
5. `ollama_fleet/orchestrator/context_builder.py` - Context builder
6. `ollama_fleet/orchestrator/model_router.py` - Model routing

**Updated Files (1):**
1. `ollama_fleet/memory/__init__.py` - Export new classes

**Total Lines of Code Added:** ~1,500+

---

## Next Steps (Immediate)

1. **Integration**: Connect ProjectMemoryManager to StateOrchestrator
2. **Wire-up**: Connect CapabilityRegistry to AgentExecutor for actual LLM calls
3. **Testing**: Create unit tests for new components
4. **Refactor Agents**: Update existing agent code to use new context system
5. **Update Main**: Modify main orchestrator entry point to use StateOrchestrator
