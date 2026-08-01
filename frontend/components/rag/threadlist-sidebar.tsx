"use client";

import { LibraryBigIcon } from "lucide-react";
import type { FC } from "react";
import { ThreadList } from "@/components/assistant-ui/thread-list";

/**
 * Fixed left column holding the thread list. A plain flex column rather than
 * the shadcn Sidebar, which would pull in the collapsible rail, mobile sheet
 * and cookie-backed open state for no gain here.
 */
export const ThreadListSidebar: FC = () => {
  return (
    <aside
      data-slot="aui_threadlist-sidebar"
      className="bg-muted/30 hidden w-64 shrink-0 flex-col border-e md:flex"
    >
      <div className="flex h-14 items-center gap-2 border-b px-4">
        <div className="bg-primary text-primary-foreground flex aspect-square size-7 items-center justify-center rounded-lg">
          <LibraryBigIcon className="size-4" />
        </div>
        <span className="truncate text-sm font-semibold">
          Research Assistant
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2">
        <ThreadList />
      </div>
    </aside>
  );
};
