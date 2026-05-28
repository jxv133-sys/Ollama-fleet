from __future__ import annotations

import logging
import time
from typing import Any

from ollama_fleet.tools.file_tools import FileTools
from ollama_fleet.tools.git_tools import GitTools
from ollama_fleet.tools.shell_tools import ShellTools

logger = logging.getLogger(__name__)


class ToolRuntime:
    def __init__(self, workspace_root: str, tools_config: dict[str, Any] | None = None) -> None:
        self.workspace_root = workspace_root
        self.tools_config = tools_config or {}
        self.tools = {
            "file_tools": FileTools(workspace_root),
            "shell_tools": ShellTools(workspace_root),
            "git_tools": GitTools(workspace_root, git_enabled=self.tools_config.get("git_enabled", True)),
        }

    async def invoke(self, tool_name: str, args: dict[str, Any], task_id: str | None = None) -> Any:
        start = time.monotonic()
        tool_obj = self.tools.get(tool_name)
        if tool_obj is None:
            return {"error_type": "tool_not_found", "tool_name": tool_name}

        action = args.get("action")
        try:
            if tool_name == "file_tools":
                if action == "read_file":
                    result = tool_obj.read_file(args["path"])
                elif action == "write_file":
                    result = tool_obj.write_file(args["path"], args["content"])
                elif action == "list_files":
                    result = tool_obj.list_files(args.get("path", "."))
                elif action == "search_code":
                    result = tool_obj.search_code(args.get("path", "."), args["pattern"])
                else:
                    result = {"error_type": "unsupported_action", "action": action}
            elif tool_name == "shell_tools":
                if action == "run_command":
                    result = tool_obj.run_command(args["args"], float(args.get("timeout", 60.0)))
                elif action == "run_tests":
                    result = tool_obj.run_tests(float(args.get("timeout", 120.0)))
                else:
                    result = {"error_type": "unsupported_action", "action": action}
            elif tool_name == "git_tools":
                if action == "git_diff":
                    result = tool_obj.git_diff()
                elif action == "git_commit":
                    result = tool_obj.git_commit(args["message"])
                else:
                    result = {"error_type": "unsupported_action", "action": action}
            else:
                result = {"error_type": "tool_not_found", "tool_name": tool_name}
        except Exception as exc:
            result = {"error_type": "exception", "message": str(exc)}

        duration = time.monotonic() - start
        logger.info(
            "Tool invocation: %s task_id=%s action=%s duration=%.3fs result=%s",
            tool_name,
            task_id,
            action,
            duration,
            getattr(result, "__class__", type(result)).__name__,
        )
        return result
