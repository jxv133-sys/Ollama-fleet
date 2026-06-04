"""Tests for file type detection."""

import pytest
from ollama_fleet.orchestrator.file_utils import FileTypeDetector, FileType


class TestFileTypeDetection:
    """Tests for FileTypeDetector."""

    def test_detect_python_by_extension(self):
        """Test Python file detection by extension."""
        assert FileTypeDetector.detect_from_path("main.py") == FileType.PYTHON
        assert FileTypeDetector.detect_from_path("utils.py") == FileType.PYTHON
        assert FileTypeDetector.detect_from_path("app.pyw") == FileType.PYTHON

    def test_detect_javascript_by_extension(self):
        """Test JavaScript file detection by extension."""
        assert FileTypeDetector.detect_from_path("app.js") == FileType.JAVASCRIPT
        assert FileTypeDetector.detect_from_path("module.mjs") == FileType.JAVASCRIPT

    def test_detect_typescript_by_extension(self):
        """Test TypeScript file detection by extension."""
        assert FileTypeDetector.detect_from_path("app.ts") == FileType.TYPESCRIPT
        assert FileTypeDetector.detect_from_path("component.tsx") == FileType.TYPESCRIPT

    def test_detect_json_by_extension(self):
        """Test JSON file detection by extension."""
        assert FileTypeDetector.detect_from_path("config.json") == FileType.JSON
        assert FileTypeDetector.detect_from_path("package.json") == FileType.JSON

    def test_detect_yaml_by_extension(self):
        """Test YAML file detection by extension."""
        assert FileTypeDetector.detect_from_path("config.yaml") == FileType.YAML
        assert FileTypeDetector.detect_from_path("config.yml") == FileType.YAML

    def test_detect_html_by_extension(self):
        """Test HTML file detection by extension."""
        assert FileTypeDetector.detect_from_path("index.html") == FileType.HTML

    def test_detect_css_by_extension(self):
        """Test CSS file detection by extension."""
        assert FileTypeDetector.detect_from_path("style.css") == FileType.CSS

    def test_detect_sql_by_extension(self):
        """Test SQL file detection by extension."""
        assert FileTypeDetector.detect_from_path("schema.sql") == FileType.SQL

    def test_detect_markdown_by_extension(self):
        """Test Markdown file detection by extension."""
        assert FileTypeDetector.detect_from_path("README.md") == FileType.MARKDOWN

    def test_detect_unknown_extension(self):
        """Test unknown extension returns UNKNOWN."""
        assert FileTypeDetector.detect_from_path("file.xyz") == FileType.UNKNOWN

    def test_detect_python_from_content(self):
        """Test Python detection from content."""
        python_code = """
def hello():
    return "world"
"""
        assert (
            FileTypeDetector.detect_from_content(python_code, "unknown_file")
            == FileType.PYTHON
        )

    def test_detect_json_from_content(self):
        """Test JSON detection from content."""
        json_content = '{"key": "value", "number": 42}'
        result = FileTypeDetector.detect_from_content(json_content)
        assert result == FileType.JSON

    def test_detect_html_from_content(self):
        """Test HTML detection from content."""
        html_content = "<!DOCTYPE html><html><head><title>Test</title></head></html>"
        result = FileTypeDetector.detect_from_content(html_content)
        assert result == FileType.HTML

    def test_detect_sql_from_content(self):
        """Test SQL detection from content."""
        sql_content = "SELECT * FROM users WHERE id = 1;"
        result = FileTypeDetector.detect_from_content(sql_content)
        assert result == FileType.SQL

    def test_detect_prefers_path_over_content(self):
        """Test that path detection takes precedence over content."""
        # Content looks like Python but path says JSON
        weird_content = '{"code": "def hello(): pass"}'
        result = FileTypeDetector.detect("config.json", weird_content)
        assert result == FileType.JSON

    def test_validate_python_syntax_valid(self):
        """Test Python syntax validation with valid code."""
        valid_code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        is_valid, error = FileTypeDetector._validate_python(valid_code)
        assert is_valid
        assert error == ""

    def test_validate_python_syntax_invalid(self):
        """Test Python syntax validation with invalid code."""
        invalid_code = """
def add(a, b)
    return a + b
"""
        is_valid, error = FileTypeDetector._validate_python(invalid_code)
        assert not is_valid
        assert ":" in error or "invalid" in error.lower()

    def test_validate_json_valid(self):
        """Test JSON validation with valid JSON."""
        valid_json = '{"name": "John", "age": 30}'
        is_valid, error = FileTypeDetector._validate_json(valid_json)
        assert is_valid
        assert error == ""

    def test_validate_json_invalid(self):
        """Test JSON validation with invalid JSON."""
        invalid_json = '{"name": "John", "age": 30'
        is_valid, error = FileTypeDetector._validate_json(invalid_json)
        assert not is_valid

    def test_validate_javascript_balanced_braces(self):
        """Test JavaScript validation with balanced braces."""
        code = "function test() { return 42; }"
        is_valid, error = FileTypeDetector._validate_javascript(code)
        assert is_valid

    def test_validate_javascript_unbalanced_braces(self):
        """Test JavaScript validation with unbalanced braces."""
        code = "function test() { return 42;"
        is_valid, error = FileTypeDetector._validate_javascript(code)
        assert not is_valid

    def test_get_syntax_validator_python(self):
        """Test getting Python validator."""
        validator = FileTypeDetector.get_syntax_validator(FileType.PYTHON)
        assert validator is not None
        is_valid, _ = validator("print('hello')")
        assert is_valid

    def test_get_syntax_validator_json(self):
        """Test getting JSON validator."""
        validator = FileTypeDetector.get_syntax_validator(FileType.JSON)
        assert validator is not None
        is_valid, _ = validator('{"test": true}')
        assert is_valid

    def test_get_syntax_validator_unknown(self):
        """Test getting validator for unknown type returns None."""
        validator = FileTypeDetector.get_syntax_validator(FileType.UNKNOWN)
        assert validator is None

    def test_detect_java_by_extension(self):
        """Test Java detection."""
        assert FileTypeDetector.detect_from_path("Main.java") == FileType.JAVA

    def test_detect_go_by_extension(self):
        """Test Go detection."""
        assert FileTypeDetector.detect_from_path("main.go") == FileType.GO

    def test_detect_rust_by_extension(self):
        """Test Rust detection."""
        assert FileTypeDetector.detect_from_path("main.rs") == FileType.RUST

    def test_detect_csharp_by_extension(self):
        """Test C# detection."""
        assert FileTypeDetector.detect_from_path("Program.cs") == FileType.CSHARP

    def test_detect_bash_by_extension(self):
        """Test Bash detection."""
        assert FileTypeDetector.detect_from_path("script.sh") == FileType.BASH
