"use client";

import { TimerIcon } from "lucide-react";
import type { FC } from "react";
import { useAuiState } from "@assistant-ui/react";
import { formatMs, type RagLatency } from "@/lib/rag";

const latencyOf = (custom: Record<string, unknown> | undefined) =>
  custom?.["latency"] as RagLatency | undefined;

/**
 * Shows how long the answer took, as measured in the browser by the RAG
 * adapter. Each selector returns a primitive (see MessageSources for why).
 */
export const MessageLatency: FC = () => {
  const endpoint = useAuiState(
    (s) => latencyOf(s.message.metadata.custom)?.endpoint,
  );
  const totalMs = useAuiState(
    (s) => latencyOf(s.message.metadata.custom)?.totalMs,
  );
  const firstTokenMs = useAuiState(
    (s) => latencyOf(s.message.metadata.custom)?.firstTokenMs,
  );

  if (endpoint === undefined || totalMs === undefined) return null;

  return (
    <div
      data-slot="aui_assistant-message-latency"
      className="text-muted-foreground mt-2 flex items-center gap-1.5 text-xs tabular-nums"
    >
      <TimerIcon className="size-3.5 shrink-0" aria-hidden />
      <span className="font-medium">/{endpoint}</span>
      {firstTokenMs !== undefined && (
        <>
          <span aria-hidden>·</span>
          <span>first token {formatMs(firstTokenMs)}</span>
        </>
      )}
      <span aria-hidden>·</span>
      <span>total {formatMs(totalMs)}</span>
    </div>
  );
};
