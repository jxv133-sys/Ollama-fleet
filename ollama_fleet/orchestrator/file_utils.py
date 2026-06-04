"""File utilities - type detection, path analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class FileType(str, Enum):
    """Supported file types."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    RUST = "rust"
    GO = "go"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    GROOVY = "groovy"
    BASH = "bash"
    SHELL = "shell"
    SQL = "sql"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    XML = "xml"
    HTML = "html"
    CSS = "css"
    MARKDOWN = "markdown"
    TEXT = "text"
    UNKNOWN = "unknown"


class FileTypeDetector:
    """Detect file type from path."""

    # Extension to file type mapping
    EXTENSION_MAP = {
        # Python
        ".py": FileType.PYTHON,
        ".pyw": FileType.PYTHON,
        # JavaScript/TypeScript
        ".js": FileType.JAVASCRIPT,
        ".mjs": FileType.JAVASCRIPT,
        ".ts": FileType.TYPESCRIPT,
        ".tsx": FileType.TYPESCRIPT,
        ".jsx": FileType.JAVASCRIPT,
        # Java
        ".java": FileType.JAVA,
        # C++
        ".cpp": FileType.CPP,
        ".cc": FileType.CPP,
        ".cxx": FileType.CPP,
        ".hpp": FileType.CPP,
        ".h": FileType.CPP,
        # C#
        ".cs": FileType.CSHARP,
        # Rust
        ".rs": FileType.RUST,
        # Go
        ".go": FileType.GO,
        # Ruby
        ".rb": FileType.RUBY,
        ".erb": FileType.RUBY,
        # PHP
        ".php": FileType.PHP,
        # Swift
        ".swift": FileType.SWIFT,
        # Kotlin
        ".kt": FileType.KOTLIN,
        ".kts": FileType.KOTLIN,
        # Scala
        ".scala": FileType.SCALA,
        # Groovy
        ".groovy": FileType.GROOVY,
        # Shell/Bash
        ".sh": FileType.BASH,
        ".bash": FileType.BASH,
        # SQL
        ".sql": FileType.SQL,
        # Data formats
        ".json": FileType.JSON,
        ".yaml": FileType.YAML,
        ".yml": FileType.YAML,
        ".toml": FileType.TOML,
        ".xml": FileType.XML,
        # Web
        ".html": FileType.HTML,
        ".htm": FileType.HTML,
        ".css": FileType.CSS,
        # Markdown
        ".md": FileType.MARKDOWN,
        ".markdown": FileType.MARKDOWN,
        # Text
        ".txt": FileType.TEXT,
        ".log": FileType.TEXT,
    }

    # Shebang patterns for script detection
    SHEBANG_MAP = {
        "#!/usr/bin/env python": FileType.PYTHON,
        "#!/usr/bin/python": FileType.PYTHON,
        "#!/usr/bin/env node": FileType.JAVASCRIPT,
        "#!/usr/bin/env ruby": FileType.RUBY,
        "#!/bin/bash": FileType.BASH,
        "#!/bin/sh": FileType.SHELL,
        "#!/usr/bin/env bash": FileType.BASH,
    }

    @staticmethod
    def detect_from_path(file_path: str) -> FileType:
        """Detect file type from file path.

        Args:
            file_path: Path to the file

        Returns:
            FileType enum value
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Check extension map
        if suffix in FileTypeDetector.EXTENSION_MAP:
            return FileTypeDetector.EXTENSION_MAP[suffix]

        # Check if it looks like a known type by name
        name_lower = path.name.lower()
        if name_lower in ("dockerfile", "makefile"):
            return FileType.TEXT
        if name_lower.startswith("dockerfile."):
            return FileType.TEXT

        return FileType.UNKNOWN

    @staticmethod
    def detect_from_content(content: str, file_path: str = "") -> FileType:
        """Detect file type from content analysis.

        Args:
            content: File content
            file_path: Optional path for extension fallback

        Returns:
            FileType enum value
        """
        if not content:
            return FileType.UNKNOWN

        # Try path first (more reliable)
        path_type = FileTypeDetector.detect_from_path(file_path)
        if path_type != FileType.UNKNOWN:
            return path_type

        # Check first line for shebang
        first_line = content.split("\n")[0] if content else ""
        for shebang, ftype in FileTypeDetector.SHEBANG_MAP.items():
            if first_line.startswith(shebang):
                return ftype

        # Content heuristics
        content_lower = content.lower()

        # Python indicators
        if any(
            x in content_lower for x in ["import ", "from ", "def ", "class ", "__main__"]
        ):
            if "import " in content_lower or "def " in content_lower:
                return FileType.PYTHON

        # JavaScript/TypeScript indicators
        if any(
            x in content_lower for x in ["import ", "export ", "function ", "const ", "let "]
        ):
            if "react" in content_lower or "jsx" in content_lower:
                return FileType.JAVASCRIPT
            if "typescript" in content_lower or ": string" in content:
                return FileType.TYPESCRIPT

        # JSON indicators
        if content_lower.strip().startswith(("{", "[")):
            try:
                import json

                json.loads(content)
                return FileType.JSON
            except (ValueError, json.JSONDecodeError):
                pass

        # HTML indicators
        if any(x in content_lower for x in ["<!doctype", "<html", "<head", "<body"]):
            return FileType.HTML

        # Java indicators
        if any(x in content for x in ["public class ", "private class ", "package "]):
            return FileType.JAVA

        # SQL indicators
        if any(
            x in content_lower
            for x in [
                "select ",
                "insert ",
                "update ",
                "delete ",
                "create table",
            ]
        ):
            return FileType.SQL

        return FileType.UNKNOWN

    @staticmethod
    def detect(file_path: str, content: str = "") -> FileType:
        """Detect file type using both path and content.

        Path detection is tried first (faster and more reliable),
        then content analysis is used as fallback.

        Args:
            file_path: Path to the file
            content: Optional file content for deeper analysis

        Returns:
            FileType enum value
        """
        # Try path-based detection first
        path_type = FileTypeDetector.detect_from_path(file_path)
        if path_type != FileType.UNKNOWN:
            return path_type

        # Fall back to content-based detection
        if content:
            return FileTypeDetector.detect_from_content(content, file_path)

        return FileType.UNKNOWN

    @staticmethod
    def get_syntax_validator(file_type: FileType) -> callable:
        """Get syntax validator function for file type.

        Args:
            file_type: Type of file

        Returns:
            Callable that validates syntax or None if not available
        """
        if file_type == FileType.PYTHON:
            return FileTypeDetector._validate_python
        elif file_type == FileType.JSON:
            return FileTypeDetector._validate_json
        elif file_type in (FileType.JAVASCRIPT, FileType.TYPESCRIPT):
            return FileTypeDetector._validate_javascript
        else:
            return None

    @staticmethod
    def _validate_python(content: str) -> tuple[bool, str]:
        """Validate Python syntax."""
        try:
            compile(content, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    @staticmethod
    def _validate_json(content: str) -> tuple[bool, str]:
        """Validate JSON syntax."""
        import json

        try:
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, str(e)

    @staticmethod
    def _validate_javascript(content: str) -> tuple[bool, str]:
        """Basic JavaScript validation (no parser, just structure checks)."""
        # Very basic checks
        brace_count = content.count("{") - content.count("}")
        paren_count = content.count("(") - content.count(")")
        bracket_count = content.count("[") - content.count("]")

        if brace_count != 0:
            return False, "Mismatched braces"
        if paren_count != 0:
            return False, "Mismatched parentheses"
        if bracket_count != 0:
            return False, "Mismatched brackets"

        return True, ""
