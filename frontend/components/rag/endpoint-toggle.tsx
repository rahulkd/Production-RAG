"use client";

import { ZapIcon, ZapOffIcon } from "lucide-react";
import type { FC } from "react";
import { useRagSettings } from "@/components/rag/rag-settings";

/**
 * Switches between the backend's streaming (/api/v1/stream) and blocking
 * (/api/v1/ask) endpoints, so their latency can be compared from the UI.
 */
export const EndpointToggle: FC = () => {
  const { settings, setEndpoint } = useRagSettings();
  const streaming = settings.endpoint === "stream";
  const Icon = streaming ? ZapIcon : ZapOffIcon;

  return (
    <button
      type="button"
      aria-pressed={streaming}
      aria-label="Stream the answer"
      title={
        streaming
          ? "Streaming via /api/v1/stream. Click to use /api/v1/ask."
          : "Single response via /api/v1/ask. Click to stream."
      }
      onClick={() => setEndpoint(streaming ? "ask" : "stream")}
      className="border-border/60 text-muted-foreground hover:bg-muted-foreground/10 dark:border-muted-foreground/15 flex h-7 cursor-pointer items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium transition-colors"
    >
      <Icon className="size-3.5 shrink-0" aria-hidden />
      <span className="text-foreground whitespace-nowrap">
        {streaming ? "/stream" : "/ask"}
      </span>
    </button>
  );
};
