#!/usr/bin/env python3
"""Main entry point for the project."""
from src.utils import helper_function
from src.core import CoreModule

def main():
    print("Starting application...")
    result = helper_function("test")
    core = CoreModule()
    core.process(result)

if __name__ == "__main__":
    main()
