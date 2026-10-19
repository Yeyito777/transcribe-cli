import { readFile } from "node:fs/promises";
import { basename, extname } from "node:path";
import { daemonRequest } from "./daemon";

/** Accepted and ignored: daemons before the socket hand-off spliced it into the command. */
const LEGACY_AUTH_ARG = "--exocortex-auth-openai";
const UNKNOWN_AUDIO_MIME_TYPE = "application/octet-stream";

const AUDIO_MIME_BY_EXTENSION: Record<string, string> = {
  ".aac": "audio/aac",
  ".aif": "audio/aiff",
  ".aiff": "audio/aiff",
  ".amr": "audio/amr",
  ".caf": "audio/x-caf",
  ".flac": "audio/flac",
  ".m4a": "audio/mp4",
  ".mka": "audio/x-matroska",
  ".mp3": "audio/mpeg",
  ".mp4": "audio/mp4",
  ".mpeg": "audio/mpeg",
  ".oga": "audio/ogg",
  ".ogg": "audio/ogg",
  ".opus": "audio/ogg",
  ".wav": "audio/wav",
  ".wave": "audio/wav",
  ".weba": "audio/webm",
  ".webm": "audio/webm",
};

function usage(): string {
  return `usage: transcribe [options] <audio-file>\n\noptions:\n  --mime-type <type>   Override inferred MIME type\n  --json               Print JSON instead of plain transcript text\n  -h, --help           Show this help\n\nexocortexd transcribes the file with its own OpenAI login.`;
}

function die(message: string, code = 1): never {
  console.error(`transcribe: ${message}`);
  process.exit(code);
}

export function parseArgs(argv: string[]): { filePath: string; mimeType: string | null; json: boolean } {
  let mimeType: string | null = null;
  let json = false;
  const positionals: string[] = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === LEGACY_AUTH_ARG) {
      i++;
    } else if (arg.startsWith(`${LEGACY_AUTH_ARG}=`)) {
      continue;
    } else if (arg === "--mime-type" || arg === "--mime") {
      mimeType = argv[++i] ?? null;
      if (!mimeType) die(`${arg} requires a value`);
    } else if (arg.startsWith("--mime-type=")) {
      mimeType = arg.slice("--mime-type=".length);
      if (!mimeType) die("--mime-type requires a value");
    } else if (arg === "--json") {
      json = true;
    } else if (arg === "-h" || arg === "--help" || arg === "help") {
      console.log(usage());
      process.exit(0);
    } else {
      positionals.push(arg);
    }
  }

  if (positionals.length !== 1) die("expected exactly one audio file path");
  return { filePath: positionals[0]!, mimeType, json };
}

function inferAudioMimeType(filePath: string): string | null {
  return AUDIO_MIME_BY_EXTENSION[extname(filePath).toLowerCase()] ?? null;
}

async function transcribe(bytes: Uint8Array, mimeType: string, filename: string): Promise<string> {
  if (bytes.byteLength === 0) throw new Error("Audio file is empty");
  const result = await daemonRequest<{ text?: unknown }>(
    { type: "transcribe_audio", audioBase64: Buffer.from(bytes).toString("base64"), mimeType, filename },
    "transcription_result",
  );
  const text = typeof result.text === "string" ? result.text.trim() : "";
  if (!text) throw new Error("transcription returned an empty result");
  return text;
}

async function main(): Promise<void> {
  const { filePath, mimeType, json } = parseArgs(process.argv.slice(2));
  const bytes = await readFile(filePath);
  const effectiveMimeType = (mimeType?.trim() || inferAudioMimeType(filePath) || UNKNOWN_AUDIO_MIME_TYPE);
  const text = await transcribe(bytes, effectiveMimeType, basename(filePath));
  if (json) console.log(JSON.stringify({ text, mimeType: effectiveMimeType }, null, 2));
  else console.log(text);
}

if (import.meta.main) main().catch((err) => die(err instanceof Error ? err.message : String(err)));
