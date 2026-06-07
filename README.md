# Agnescli

[![CI](https://github.com/makur6371/agnescli/actions/workflows/ci.yml/badge.svg)](https://github.com/makur6371/agnescli/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/agnescli)](https://pypi.org/project/agnescli/)
[![Python](https://img.shields.io/pypi/pyversions/agnescli)](https://pypi.org/project/agnescli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Agnes AI** - The world's first unlimited free multimodal API. Chat, image generation, and video generation, all under one API key, no usage limits.

Agnescli is the autonomous agent CLI built for [Agnes AI](https://agnes-ai.com). It combines language understanding, image generation, and video generation into a single interactive terminal experience. Ask it to write code, generate images, create videos, manage files, or chain them all together - it plans, executes, and delivers.

## Why Agnes AI?

- **Free & Unlimited** - No usage caps, no credit card required
- **Multimodal** - Text, image, and video models under one API
- **Simple** - One API key for everything

## Install

```bash
pip install agnescli
```

Or from source:

```bash
git clone https://github.com/makur6371/agnescli.git
cd agnescli
pip install -e .
```

## Quick Start

```bash
# 1. Get your free API key at https://agnes-ai.com
# 2. Save it
agnescli setup YOUR_API_KEY

# 3. Start using
agnescli
```

Or set the environment variable: `export AGNES_API_KEY=your_key`

## Usage

```bash
agnescli              # interactive agent
agnescli -y           # auto-confirm all tool calls
agnescli -p "prompt"  # non-interactive mode, run and exit
agnescli -c           # continue last session
agnescli resume       # list and resume sessions
agnescli models       # list available models
```

## What Can It Do?

**Chat & Code** - Ask questions, write code, debug issues, explain concepts.

**Generate Images** - `Generate a cyberpunk cityscape at sunset` - the agent calls Agnes Image AI and returns the result.

**Generate Videos** - `Create a short video of ocean waves` - the agent submits the job, polls for progress, and delivers the video.

**File Operations** - Read, write, edit files, search codebases, run shell commands - all with built-in path protection.

**Plan & Execute** - `/plan build a to-do app with React` - the agent breaks it into steps, confirms with you, then executes each one.

## Slash Commands

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

## Agent Tools

The agent has access to 10 built-in tools:

| Tool | Description |
|------|-------------|
| `shell` | Execute shell commands |
| `read_file` | Read file contents |
| `write_file` | Write/create files (with path protection) |
| `edit_file` | Precise string replacement |
| `list_files` | List directory contents |
| `glob` | Find files by pattern (e.g. `**/*.py`) |
| `grep` | Search file contents with regex |
| `python_exec` | Run Python code |
| `generate_image` | Text-to-image via Agnes Image AI |
| `generate_video` | Text-to-video via Agnes Video AI |

## Safety

**Path Protection** - Write operations to sensitive paths are blocked automatically:
- `.git`, `.env`, `.ssh`, `.gnupg`, `.aws`, etc.
- Root directories (`/`, `C:\`)
- Credential files (`id_rsa`, `credentials.json`, etc.)

**Tool Confirmation** - By default, the agent asks before executing tools. Use `-y` to auto-confirm.

## Models

| Model | Type | Context |
|-------|------|---------|
| agnes-2.0-flash | Chat / Language | 256K |
| agnes-image-2.1-flash | Image Generation | - |
| agnes-video-v2.0 | Video Generation | 441 frames |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT
