# CallerAgent

A Voice AI system that captures microphone audio, detects speech, transcribes it with Groq Whisper,
generates conversational replies with Groq GPT‑OSS, and plays responses using Groq PlayAI TTS.

This repository contains the application code, configuration, and Docker files needed to run
the CallerAgent locally or inside a container.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the application](#running-the-application)
  - [Using uv (recommended)](#using-uv-recommended)
  - [Without uv (plain Python)](#without-uv-plain-python)
- [Docker](#docker)
- [Logging](#logging)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- Real-time microphone capture and voice activity detection (VAD)
- Speech-to-text transcription via Groq Whisper
- Conversational AI responses via Groq GPT‑OSS
- Text-to-speech synthesis via Groq PlayAI TTS
- Config-driven models and audio parameters


## Prerequisites

- Python 3.10+ (the project contains a virtual environment under `env/` but using a clean venv is recommended)
- A working microphone and speakers on the host machine
- A Groq API key with permissions for Whisper, GPT, and PlayAI (set via environment variable `GROQ_API_KEY`)
- Optional: `uv` tool for managing and running the project (see `pyproject.toml` and `uv.lock`)

## Installation

1. Clone the repository:

```powershell
git clone https://github.com/SABARNO-PRAMANICK/CallerAgent.git
cd CallerAgent
```
2. Create and activate a virtual environment:

```powershell
uv venv .venv:
```
3. Install dependencies:

```powershell
uv pip install -r [requirements.txt]
```

4. Configuration:
```
GROQ_API_KEY=your_groq_api_key_here
# Optional overrides (examples)
#SAMPLE_RATE=16000
#CHUNK_SIZE=2048
```

5. Run the application:

```powershell
uv run python -m app.main
```
