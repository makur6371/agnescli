"""Built-in tools for the Agnescli agent."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import AgnesClient

# ── Path protection ───────────────────────────────────────────────────

PROTECTED_PREFIXES = [
    ".git",
    ".env",
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".config",
    ".npmrc",
    ".pypirc",
    ".docker",
    ".kube",
    "node_modules",
]

PROTECTED_EXACT = [
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.json",
    "token.json",
]


def check_path_protection(path: str) -> str | None:
    """Return error message if path is protected, None if safe."""
    p = Path(path).resolve()
    name = p.name
    parts = p.parts

    # Block writing to root directories (C:\, /, etc.)
    if len(p.parts) <= 2:
        return f"Cannot write to root directory: {path} (use a project subdirectory)"

    for protected in PROTECTED_EXACT:
        if name == protected:
            return f"Protected file: {path} (contains sensitive data)"

    for part in parts:
        for prefix in PROTECTED_PREFIXES:
            if part == prefix or part.startswith(prefix + "."):
                return f"Protected path: {path} (contains '{part}')"
    return None


# ── Tool definitions (OpenAI function calling format) ──────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a shell command and return stdout/stderr. Use this to run any system command, install packages, build projects, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "workdir": {"type": "string", "description": "Working directory (optional)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the full text content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {"type": "integer", "description": "Line number to start from (0-based, optional)"},
                    "limit": {"type": "integer", "description": "Max lines to read (optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories as needed. Cannot write to protected paths (.git, .env, .ssh, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file. Safer than write_file - only changes the specified text. The old_string must appear exactly once in the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find and replace (must appear exactly once)",
                    },
                    "new_string": {"type": "string", "description": "Text to replace it with"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list (default: current directory)"},
                    "pattern": {"type": "string", "description": "Glob pattern to filter, e.g. '*.py' (optional)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern recursively. Returns matching file paths. Example patterns: '**/*.py', 'src/**/*.ts', '*.md'",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match (e.g. '**/*.py')"},
                    "path": {
                        "type": "string",
                        "description": "Root directory to search from (default: current directory)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers. Supports regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default: current directory)",
                    },
                    "glob": {"type": "string", "description": "File pattern to filter, e.g. '*.py' (optional)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute a Python code snippet and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text prompt using Agnes Image AI. Returns the image URL. Use this when the user asks to create, generate, or draw an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "size": {"type": "string", "description": "Image size, e.g. '1024x768' (default)"},
                    "input_image_url": {
                        "type": "string",
                        "description": "URL of input image for image-to-image (optional)",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": "Generate a video from a text prompt using Agnes Video AI. This is an async task - it submits a job and polls until complete. Returns the video URL. Use this when the user asks to create or generate a video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the video to generate"},
                    "input_image_url": {
                        "type": "string",
                        "description": "URL of input image for image-to-video (optional)",
                    },
                    "width": {"type": "integer", "description": "Video width (default: 1152)"},
                    "height": {"type": "integer", "description": "Video height (default: 768)"},
                    "num_frames": {"type": "integer", "description": "Number of frames, 8n+1, max 441 (default: 121)"},
                    "frame_rate": {"type": "integer", "description": "FPS, 1-60 (default: 24)"},
                },
                "required": ["prompt"],
            },
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────


def execute_tool(name: str, arguments: dict[str, Any], client: AgnesClient) -> str:
    """Execute a tool by name and return the result as a string."""
    try:
        match name:
            case "shell":
                return _shell(**arguments)
            case "read_file":
                return _read_file(**arguments)
            case "write_file":
                return _write_file(**arguments)
            case "edit_file":
                return _edit_file(**arguments)
            case "list_files":
                return _list_files(**arguments)
            case "glob":
                return _glob(**arguments)
            case "grep":
                return _grep(**arguments)
            case "python_exec":
                return _python_exec(**arguments)
            case "generate_image":
                return _generate_image(client, **arguments)
            case "generate_video":
                return _generate_video(client, **arguments)
            case _:
                return f"Error: Unknown tool '{name}'"
    except Exception as e:
        return f"Error executing {name}: {e}"


# ── Shell ──────────────────────────────────────────────────────────────


def _shell(command: str, workdir: str | None = None, timeout: int = 60) -> str:
    is_windows = sys.platform == "win32"
    shell_cmd = ["powershell", "-NoProfile", "-Command", command] if is_windows else ["bash", "-c", command]
    try:
        result = subprocess.run(shell_cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir)
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        parts.append(f"[exit code: {result.returncode}]")
        output = "\n".join(parts)
        if len(output) > 8000:
            output = output[:8000] + "\n... (output truncated)"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"


# ── File operations ────────────────────────────────────────────────────


def _read_file(path: str, offset: int | None = None, limit: int | None = None) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    if p.is_dir():
        return f"Error: {path} is a directory, not a file. Use list_files instead."
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: {path} is a binary file and cannot be read as text."

    lines = text.splitlines(keepends=True)
    start = offset or 0
    end = start + limit if limit else len(lines)
    selected = lines[start:end]

    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i:>4} | {line.rstrip()}")
    result = "\n".join(numbered)
    if len(result) > 8000:
        result = result[:8000] + "\n... (content truncated)"
    return result


def _write_file(path: str, content: str) -> str:
    err = check_path_protection(path)
    if err:
        return f"Error: {err}"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {path}"


def _edit_file(path: str, old_string: str, new_string: str) -> str:
    err = check_path_protection(path)
    if err:
        return f"Error: {err}"
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: {path} is a binary file."

    count = text.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path}"
    if count > 1:
        return f"Error: old_string appears {count} times in {path} (must be unique). Provide more context."

    new_text = text.replace(old_string, new_string, 1)
    p.write_text(new_text, encoding="utf-8")
    return f"Successfully edited {path}"


def _list_files(path: str = ".", pattern: str | None = None) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: Path not found: {path}"
    if not p.is_dir():
        return f"Error: {path} is not a directory"

    entries = sorted(p.glob(pattern)) if pattern else sorted(p.iterdir())
    if not entries:
        return f"(empty directory: {path})"

    lines = []
    for entry in entries[:100]:
        if entry.is_dir():
            lines.append(f"  [dir]  {entry.name}/")
        else:
            size = entry.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            lines.append(f"  [file] {size_str:>8}  {entry.name}")

    if len(entries) > 100:
        lines.append(f"  ... and {len(entries) - 100} more entries")
    return "\n".join(lines)


def _glob(pattern: str, path: str = ".") -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: Path not found: {path}"
    matches = sorted(p.glob(pattern))
    if not matches:
        return f"No files matching '{pattern}' in {path}"
    lines = []
    for m in matches[:200]:
        try:
            rel = m.relative_to(Path.cwd())
            lines.append(str(rel))
        except ValueError:
            lines.append(str(m))
    if len(matches) > 200:
        lines.append(f"... and {len(matches) - 200} more")
    return "\n".join(lines)


def _grep(pattern: str, path: str = ".", glob: str | None = None) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: Path not found: {path}"

    regex = re.compile(pattern)

    files = []
    if p.is_file():
        files = [p]
    else:
        if glob:
            files = sorted(p.rglob(glob))
        else:
            files = sorted(p.rglob("*"))
        files = [f for f in files if f.is_file()]

    results = []
    for fpath in files[:500]:
        try:
            text = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                try:
                    rel = fpath.relative_to(Path.cwd())
                    results.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                except ValueError:
                    results.append(f"{fpath}:{i}: {line.rstrip()[:200]}")
                if len(results) >= 100:
                    break
        if len(results) >= 100:
            break

    if not results:
        return f"No matches for '{pattern}' in {path}"
    output = "\n".join(results)
    if len(results) >= 100:
        output += "\n... (showing first 100 matches)"
    return output


def _python_exec(code: str) -> str:
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            parts.append(f"[exit code: {result.returncode}]")
        output = "\n".join(parts)
        if len(output) > 8000:
            output = output[:8000] + "\n... (output truncated)"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Python execution timed out after 30s"


# ── Image generation ───────────────────────────────────────────────────


def _generate_image(
    client: AgnesClient,
    prompt: str,
    size: str = "1024x768",
    input_image_url: str | None = None,
) -> str:
    input_images = [input_image_url] if input_image_url else None
    result = client.image_generate(prompt, size=size, input_images=input_images)

    data = result.get("data", [])
    if not data:
        return "Error: No image returned from API"

    url = data[0].get("url", "")
    if not url:
        return "Error: No image URL in response"

    return f"Image generated successfully!\nURL: {url}"


# ── Video generation ───────────────────────────────────────────────────


def _generate_video(
    client: AgnesClient,
    prompt: str,
    input_image_url: str | None = None,
    width: int = 1152,
    height: int = 768,
    num_frames: int = 121,
    frame_rate: int = 24,
) -> str:
    result = client.video_create(
        prompt,
        image=input_image_url,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
    )

    task_id = result.get("task_id") or result.get("id", "")
    if not task_id:
        return "Error: No task ID returned"

    elapsed = 0
    while True:
        time.sleep(5)
        elapsed += 5
        status = client.video_status(task_id)
        st = status.get("status", "")
        progress = status.get("progress", 0)

        if st == "completed":
            url = status.get("video_url") or status.get("remixed_from_video_id", "")
            return f"Video generated successfully! ({elapsed}s)\nURL: {url}"
        if st == "failed":
            return f"Error: Video generation failed - {status.get('error', 'unknown')}"
        if elapsed > 300:
            return f"Error: Video generation timed out after {elapsed}s (task: {task_id})"

        print(f"  [video] {progress}% - {st} ({elapsed}s)", file=sys.stderr)
