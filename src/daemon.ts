/**
 * Minimal exocortexd socket client. Provider calls (OpenAI image generation
 * and transcription) are made by the daemon with its own login, so this tool
 * never holds a credential.
 */
import { connect } from "node:net";
import { join, resolve } from "node:path";

/** EXOCORTEX_SOCKET (set for every daemon child) or the main checkout's socket. */
export function daemonSocketPath(env: NodeJS.ProcessEnv = process.env): string {
  const explicit = env.EXOCORTEX_SOCKET?.trim();
  if (explicit) return explicit;
  const configRoot = env.EXOCORTEX_CONFIG_DIR?.trim()
    ? resolve(env.EXOCORTEX_CONFIG_DIR)
    : join(resolve(import.meta.dir, "../../.."), "config");
  return join(configRoot, "runtime", "exocortexd.sock");
}

/** Send one command and resolve with the event named `resultType` for its reqId. */
export function daemonRequest<T extends Record<string, unknown>>(
  command: Record<string, unknown>,
  resultType: string,
  options: { socketPath?: string; timeoutMs?: number } = {},
): Promise<T> {
  const socketPath = options.socketPath ?? daemonSocketPath();
  const reqId = `${command.type}-${process.pid}-${Date.now()}`;
  return new Promise<T>((resolvePromise, reject) => {
    let settled = false;
    let pending: Buffer[] = [];
    const socket = connect(socketPath);
    const finish = (error: Error | null, value?: T) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      socket.destroy();
      if (error) reject(error);
      else resolvePromise(value!);
    };
    const timeoutMs = options.timeoutMs ?? 600_000;
    const timer = setTimeout(() => finish(new Error(`exocortexd did not answer within ${Math.round(timeoutMs / 1000)}s`)), timeoutMs);
    socket.on("connect", () => socket.write(`${JSON.stringify({ ...command, reqId })}\n`));
    socket.on("data", (chunk: Buffer) => {
      // Results can be one multi-megabyte line: scan each chunk only once.
      let start = 0;
      let newline: number;
      while ((newline = chunk.indexOf(10, start)) !== -1) {
        pending.push(chunk.subarray(start, newline));
        start = newline + 1;
        const line = Buffer.concat(pending).toString("utf8");
        pending = [];
        let event: Record<string, unknown>;
        try { event = JSON.parse(line) as Record<string, unknown>; } catch { continue; }
        if (event.reqId !== reqId) continue;
        if (event.type === resultType) finish(null, event as T);
        else if (event.type === "error") finish(new Error(String(event.message ?? "exocortexd returned an error")));
      }
      if (start < chunk.length) pending.push(chunk.subarray(start));
    });
    socket.on("error", (err: NodeJS.ErrnoException) => finish(new Error(
      err.code === "ENOENT" || err.code === "ECONNREFUSED"
        ? `cannot reach exocortexd at ${socketPath}; is the daemon running?`
        : `exocortexd connection failed: ${err.message}`,
    )));
    socket.on("close", () => finish(new Error("exocortexd closed the connection before answering")));
  });
}
