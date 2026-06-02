from pathlib import Path

path = Path('ollama_fleet/agents/executor.py')
text = path.read_text()
start_marker = '    def _strip_code_fences(self, text: str) -> str:'
end_marker = '    def _normalize_critic_output(self, data: dict[str, Any]) -> dict[str, Any]:\n'
start = text.index(start_marker)
end = text.index(end_marker)
replacement = ''.join([
    '    def _strip_code_fences(self, text: str) -> str:\n',
    '        """Remove markdown fenced code blocks and return the inner code."""\n',
    '        text = text.strip()\n',
    '        if text.startswith("```"):\n',
    "            fence_match = re.search(r'^```(?:\\w+)?\\n(.*)```$', text, re.DOTALL)\n",
    '            if fence_match:\n',
    '                return fence_match.group(1).strip()\n',
    "        block_match = re.search(r'```(?:\\w+)?\\n(.*?)```', text, re.DOTALL)\n",
    '        if block_match:\n',
    '            return block_match.group(1).strip()\n',
    '        return text\n\n',
    '    def _normalize_coder_output(self, raw: str) -> str:\n',
    '        """Normalize Coder output: extract the file contents from the model response."""\n',
    '        return coder_module.normalize_coder_response(raw)\n\n',
])
text = text[:start] + replacement + text[end:]
path.write_text(text)
print('replaced')
