import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const cli = resolve(import.meta.dir, "../bin/transcribe");

function fakeDaemon(socketPath: string, reply: (command: Record<string, unknown>) => Record<string, unknown>, seen: Record<string, unknown>[]) {
  const chunks: Buffer[] = [];
  return Bun.listen({
    unix: socketPath,
    socket: {
      data(socket, data) {
        chunks.push(Buffer.from(data));
        const text = Buffer.concat(chunks).toString("utf8");
        if (!text.endsWith("\n")) return;
        const command = JSON.parse(text.trim()) as Record<string, unknown>;
        seen.push(command);
        socket.write(`${JSON.stringify({ type: "ack", reqId: command.reqId })}\n`);
        socket.write(`${JSON.stringify({ reqId: command.reqId, ...reply(command) })}\n`);
      },
    },
  });
}

describe("transcribe through exocortexd", () => {
  test("sends the audio over the daemon socket and prints the text, with no credential anywhere", async () => {
    const dir = mkdtempSync(join(tmpdir(), "transcribe-cli-test-"));
    const socketPath = join(dir, "d.sock");
    const audio = join(dir, "note.ogg");
    writeFileSync(audio, Buffer.from("OggS fake audio"));
    const seen: Record<string, unknown>[] = [];
    const server = fakeDaemon(socketPath, () => ({ type: "transcription_result", text: " hello there " }), seen);
    try {
      // A pre-socket daemon still splices this; it must be ignored, not used.
      const child = Bun.spawn([cli, "--exocortex-auth-openai", "stale", audio], {
        stdout: "pipe", stderr: "pipe", env: { ...process.env, EXOCORTEX_SOCKET: socketPath },
      });
      const [code, stdout] = await Promise.all([child.exited, new Response(child.stdout).text()]);
      expect(code).toBe(0);
      expect(stdout).toBe("hello there\n");
      expect(seen).toEqual([{
        type: "transcribe_audio", reqId: expect.any(String), mimeType: "audio/ogg", filename: "note.ogg",
        audioBase64: Buffer.from("OggS fake audio").toString("base64"),
      }]);
    } finally {
      server.stop(true);
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("reports a daemon error and a missing daemon", async () => {
    const dir = mkdtempSync(join(tmpdir(), "transcribe-cli-test-"));
    const socketPath = join(dir, "d.sock");
    const audio = join(dir, "a.wav");
    writeFileSync(audio, Buffer.from("RIFF"));
    const server = fakeDaemon(socketPath, () => ({ type: "error", message: "Voice transcription failed: Not logged in" }), []);
    try {
      const failed = Bun.spawn([cli, audio], { stdout: "pipe", stderr: "pipe", env: { ...process.env, EXOCORTEX_SOCKET: socketPath } });
      const [code, stderr] = await Promise.all([failed.exited, new Response(failed.stderr).text()]);
      expect(code).toBe(1);
      expect(stderr).toContain("transcribe: Voice transcription failed: Not logged in");
      const missing = Bun.spawn([cli, audio], { stdout: "pipe", stderr: "pipe", env: { ...process.env, EXOCORTEX_SOCKET: join(dir, "none.sock") } });
      const [code2, stderr2] = await Promise.all([missing.exited, new Response(missing.stderr).text()]);
      expect(code2).toBe(1);
      expect(stderr2).toContain("cannot reach exocortexd");
    } finally {
      server.stop(true);
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
