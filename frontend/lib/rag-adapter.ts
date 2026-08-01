import type {
  ChatModelAdapter,
  SourceMessagePart,
  ThreadMessage,
} from "@assistant-ui/react";
import { sourceTitle, type RagSettings } from "@/lib/rag";

/** One decoded `data:` frame from the backend's /api/v1/stream response. */
type StreamFrame = {
  sources?: string[];
  chunks_used?: number;
  search_mode?: string;
  chunk?: string;
  answer?: string;
  done?: boolean;
  error?: string;
};

const textOf = (message: ThreadMessage | undefined) =>
  (message?.content ?? [])
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("")
    .trim();

const toSourcePart = (url: string, index: number): SourceMessagePart => ({
  type: "source",
  sourceType: "url",
  id: `source-${index}`,
  url,
  title: sourceTitle(url),
});

const decodeFrame = (raw: string): StreamFrame | null => {
  const data = raw
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("");
  if (!data) return null;
  try {
    return JSON.parse(data) as StreamFrame;
  } catch {
    return null;
  }
};

async function* readFrames(body: ReadableStream<Uint8Array>) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder
        .decode(value, { stream: true })
        .replaceAll("\r\n", "\n");

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = decodeFrame(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (frame) yield frame;
        boundary = buffer.indexOf("\n\n");
      }
    }

    const trailing = decodeFrame(buffer);
    if (trailing) yield trailing;
  } finally {
    void reader.cancel().catch(() => {});
  }
}

const errorFrom = async (response: Response) => {
  const body = await response.text().catch(() => "");
  try {
    const parsed = JSON.parse(body) as { detail?: string; error?: string };
    if (parsed.detail || parsed.error) return parsed.detail ?? parsed.error!;
  } catch {
    // fall through to the raw body
  }
  return body.trim() || `Request failed with status ${response.status}`;
};

/**
 * Streams answers from the FastAPI RAG pipeline.
 *
 * `getSettings` is read at run time rather than captured so that changing the
 * model in the UI takes effect on the next message without rebuilding the runtime.
 */
export const createRagAdapter = (
  getSettings: () => RagSettings,
): ChatModelAdapter => ({
  async *run({ messages, abortSignal }) {
    const query = textOf(messages.at(-1));
    if (!query) return;

    const settings = getSettings();
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: abortSignal,
      body: JSON.stringify({
        query,
        model: settings.model,
        top_k: settings.topK,
        use_hybrid: settings.useHybrid,
        categories: settings.categories ?? null,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(await errorFrom(response));
    }

    let text = "";
    let sources: SourceMessagePart[] = [];

    for await (const frame of readFrames(response.body)) {
      if (frame.error) throw new Error(frame.error);

      if (frame.sources) {
        sources = frame.sources.map(toSourcePart);
      }

      if (frame.chunk) {
        text += frame.chunk;
      }

      // The backend only sends `answer` on the terminal frame. It repeats the
      // full text, which is the only content we get when nothing streamed.
      if (frame.done && !text && frame.answer) {
        text = frame.answer;
      }

      if (frame.done) {
        yield {
          content: [{ type: "text", text }, ...sources],
          status: { type: "complete", reason: "stop" },
        };
        return;
      }

      if (frame.chunk || frame.sources) {
        yield { content: [{ type: "text", text }, ...sources] };
      }
    }

    // Upstream closed without a `done` frame; keep whatever streamed.
    yield {
      content: [{ type: "text", text }, ...sources],
      status: { type: "complete", reason: "unknown" },
    };
  },
});
