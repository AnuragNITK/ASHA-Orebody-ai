"""Pipeline error types."""
from __future__ import annotations


class MineDesignError(Exception):
    """Base class for all pipeline errors."""


class IngestionError(MineDesignError):
    """A source CSV could not be loaded or mapped to the required schema."""


class SolidValidationError(MineDesignError):
    """The triangulated solid failed validation after all retries."""


class LLMError(MineDesignError):
    """An LLM call failed or returned unusable output."""


class PitOptimizationError(MineDesignError):
    """The pit optimizer could not produce a shell."""
