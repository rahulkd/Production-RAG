import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const RAG_API_URL = process.env.RAG_API_URL ?? "http://localhost:8000";

/** Server-side proxy for answer feedback; see app/api/chat/route.ts for why. */
export async function POST(request: NextRequest) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${RAG_API_URL}/api/v1/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: request.signal,
    });
  } catch (error) {
    console.error("RAG API unreachable:", error);
    return Response.json(
      { error: "The RAG API is unreachable. Is the api service running?" },
      { status: 502 },
    );
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
