# transcribe-cli

Exocortex external tool for OpenAI-backed audio transcription.

## Usage

```bash
transcribe audio.wav
transcribe audio.wav --mime-type audio/wav
```

exocortexd transcribes the audio with its own OpenAI login: the tool sends the
file over the daemon socket (`EXOCORTEX_SOCKET`, else the checkout's
`config/runtime/exocortexd.sock`) and prints the text. It never sees a
credential, and it works from any shell, including Claude Code's.
