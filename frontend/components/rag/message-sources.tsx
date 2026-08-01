"use client";

import { FileTextIcon } from "lucide-react";
import type { FC } from "react";
import { useAuiState } from "@assistant-ui/react";
import { useShallow } from "zustand/shallow";
import { sourceTitle } from "@/lib/rag";

/**
 * Renders the arXiv papers the RAG pipeline retrieved for this answer. The
 * backend sends them ahead of the tokens, so they are appended to the message
 * as `source` parts rather than shown inline with the text.
 *
 * The selector must return primitives. `useShallow` compares elements with
 * Object.is, so returning freshly built objects would make every snapshot look
 * changed and spin React into an infinite render.
 */
export const MessageSources: FC = () => {
  const urls = useAuiState(
    useShallow((s) =>
      s.message.parts.flatMap((part) =>
        part.type === "source" && part.sourceType === "url" ? [part.url] : [],
      ),
    ),
  );

  if (urls.length === 0) return null;

  return (
    <div className="aui-message-sources mt-3 flex flex-wrap items-center gap-1.5">
      <span className="text-muted-foreground me-0.5 text-xs font-medium">
        Sources
      </span>
      {urls.map((url) => (
        <a
          key={url}
          href={url}
          target="_blank"
          rel="noreferrer noopener"
          className="border-border/60 text-muted-foreground hover:bg-muted hover:text-foreground dark:border-muted-foreground/15 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors"
        >
          <FileTextIcon className="size-3.5 shrink-0" aria-hidden />
          {sourceTitle(url)}
        </a>
      ))}
    </div>
  );
};
