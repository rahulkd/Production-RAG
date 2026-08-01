import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const RAG_API_URL = process.env.RAG_API_URL ?? "http://localhost:8000";

/**
 * Server-side proxy to the FastAPI RAG pipeline.
 *
 * The browser only ever talks to this origin, so the API needs no CORS config
 * and no published port — it stays reachable only on the internal network.
 */
export async function POST(request: NextRequest) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${RAG_API_URL}/api/v1/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: request.signal,
    });
  } catch (error) {
    if (request.signal.aborted) return new Response(null, { status: 499 });
    console.error("RAG API unreachable:", error);
    return Response.json(
      { error: "The RAG API is unreachable. Is the api service running?" },
      { status: 502 },
    );
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return Response.json(
      { error: detail.trim() || `RAG API returned ${upstream.status}` },
      { status: upstream.status === 200 ? 502 : upstream.status },
    );
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      // Tells nginx/ALB-style proxies not to buffer the stream.
      "X-Accel-Buffering": "no",
    },
  });
}
