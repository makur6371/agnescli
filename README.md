# Agnescli

Autonomous Agent CLI for [Agnes AI](https://agnes-ai.com). One API key, full multimodal capability - chat, code, image, and video generation.

## Install

```bash
pip install -e .
```

## Setup

```bash
agnescli setup YOUR_API_KEY
```

Key is saved to `~/.agnescli/config.json`. Alternatively, set the `AGNES_API_KEY` environment variable.

## Usage

```bash
agnescli              # interactive agent
agnescli -y           # auto-confirm all tool calls
agnescli -p "prompt"  # non-interactive mode, run and exit
agnescli -c           # continue last session
agnescli resume       # list and resume sessions
agnescli models       # list available models
```

### Slash Commands

Type `/` to see autocomplete suggestions.

| Command | Description |
|---------|-------------|
| `/plan task` | Plan then execute step by step |
| `/image prompt` | Generate an image |
| `/video prompt` | Generate a video |
| `/thinking` | Toggle thinking mode |
| `/auto` | Toggle auto-confirm |
| `/compact` | Compress conversation context |
| `/new` | New session |
| `/resume [id]` | Resume a session |
| `/sessions` | List saved sessions |
| `/config` | Show current config |
| `/status` | Quick status |
| `/save [file]` | Save chat history |
| `/clear` | Reset conversation |
| `/help` | Show help |
| `/exit` | Quit |

### Agent Tools

The agent has access to 10 built-in tools:

- **shell** - Execute shell commands
- **read_file** - Read file contents
- **write_file** - Write/create files (with path protection)
- **edit_file** - Precise string replacement in files
- **list_files** - List directory contents
- **glob** - Find files by pattern (e.g. `**/*.py`)
- **grep** - Search file contents with regex
- **python_exec** - Run Python code
- **generate_image** - Text-to-image via Agnes Image AI
- **generate_video** - Text-to-video via Agnes Video AI

### Path Protection

Write operations to sensitive paths are blocked automatically:
- `.git`, `.env`, `.ssh`, `.gnupg`, `.aws`, etc.
- Root directories (`/`, `C:\`)
- Credential files (`id_rsa`, `credentials.json`, etc.)

### Context Management

- Automatic context compression when conversation exceeds ~80K tokens
- Manual compression via `/compact`
- Session auto-save and resume capability

## Models

| Model | Type | Context |
|-------|------|---------|
| agnes-2.0-flash | Chat / Language | 256K |
| agnes-image-2.1-flash | Image Generation | - |
| agnes-video-v2.0 | Video Generation | 441 frames |

## License

MIT
