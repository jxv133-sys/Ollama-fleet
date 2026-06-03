"""Model Router - Select optimal model for each task.

Different capabilities should use different models:
- Planning: reasoning model (e.g., claude-opus)
- Code generation: coding model (e.g., claude-sonnet)
- Code review: reasoning model (e.g., claude-opus)
- Testing analysis: reasoning model (e.g., claude-opus)

This allows cost/performance optimization and specialized models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ollama_fleet.config import FleetSettings
from ollama_fleet.agents.capabilities import CapabilityType

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model types available."""

    REASONING = "reasoning"  # e.g., claude-opus
    CODING = "coding"  # e.g., claude-sonnet
    FAST = "fast"  # e.g., claude-haiku (for simple tasks)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_name: str
    model_type: ModelType
    max_tokens: int
    temperature: float
    timeout_seconds: int


class ModelRouter:
    """Routes capability requests to appropriate models.

    Decisions based on:
    - Capability type (what we're doing)
    - Task complexity (implicit)
    - Cost/performance constraints
    """

    def __init__(self, settings: FleetSettings) -> None:
        self._settings = settings
        self._routing_table: dict[CapabilityType, ModelType] = {
            CapabilityType.PLANNING: ModelType.REASONING,
            CapabilityType.CODE_GENERATION: ModelType.CODING,
            CapabilityType.CODE_REVIEW: ModelType.REASONING,
            CapabilityType.TESTING: ModelType.REASONING,
            CapabilityType.WORKSPACE_SEARCH: ModelType.FAST,
            CapabilityType.INTERFACE_EXTRACTION: ModelType.CODING,
        }

        # Initialize model configs from settings
        self._models = self._initialize_models()

    def get_model_for_capability(
        self,
        capability_type: CapabilityType,
    ) -> ModelConfig:
        """Get the appropriate model for a capability.

        Args:
            capability_type: Type of capability to execute

        Returns:
            ModelConfig to use
        """
        model_type = self._routing_table.get(
            capability_type,
            ModelType.CODING,  # Default to coding model
        )

        if model_type in self._models:
            return self._models[model_type]

        # Fallback to default if model type not configured
        logger.warning(f"Model type {model_type} not configured, using default")
        return self._models[ModelType.CODING]

    def set_routing(
        self,
        capability_type: CapabilityType,
        model_type: ModelType,
    ) -> None:
        """Override routing for a capability.

        Args:
            capability_type: Capability type
            model_type: Model type to route to
        """
        self._routing_table[capability_type] = model_type
        logger.info(f"Updated routing: {capability_type} → {model_type}")

    def _initialize_models(self) -> dict[ModelType, ModelConfig]:
        """Initialize model configurations from settings.

        Returns:
            Dict mapping ModelType to ModelConfig
        """
        # Get model names from settings
        planning_model = getattr(
            self._settings,
            "planning_model",
            "llama2",  # Default Ollama model
        )
        coding_model = getattr(
            self._settings,
            "coding_model",
            "llama2",
        )
        fast_model = getattr(
            self._settings,
            "fast_model",
            "llama2",
        )

        return {
            ModelType.REASONING: ModelConfig(
                model_name=planning_model,
                model_type=ModelType.REASONING,
                max_tokens=2048,
                temperature=0.7,
                timeout_seconds=60,
            ),
            ModelType.CODING: ModelConfig(
                model_name=coding_model,
                model_type=ModelType.CODING,
                max_tokens=4096,
                temperature=0.5,  # Lower temp for code generation
                timeout_seconds=120,
            ),
            ModelType.FAST: ModelConfig(
                model_name=fast_model,
                model_type=ModelType.FAST,
                max_tokens=512,
                temperature=0.3,  # Very low temp for deterministic output
                timeout_seconds=30,
            ),
        }

    def get_all_models(self) -> dict[ModelType, ModelConfig]:
        """Get all configured models.

        Returns:
            Dict of all model configurations
        """
        return self._models.copy()
