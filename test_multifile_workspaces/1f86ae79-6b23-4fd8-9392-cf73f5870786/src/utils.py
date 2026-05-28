"""Utility functions for the project."""

def helper_function(value: str) -> str:
    """Helper function that processes a value."""
    return f"Processed: {value}"

def validate_input(data: any) -> bool:
    """Validate input data."""
    return isinstance(data, (str, int, list))
