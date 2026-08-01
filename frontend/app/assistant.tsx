"use client";

import {
  AssistantRuntimeProvider,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { useCallback, useMemo, useState } from "react";
import { Thread } from "@/components/assistant-ui/thread";
import { RagSettingsProvider } from "@/components/rag/rag-settings";
import { ThreadListSidebar } from "@/components/rag/threadlist-sidebar";
import {
  ragThreadListAdapter,
  setRagSettings,
  useRagThreadRuntime,
} from "@/lib/rag-runtime";
import { DEFAULT_SETTINGS, type RagSettings } from "@/lib/rag";

export const Assistant = () => {
  const [settings, setSettings] = useState<RagSettings>(DEFAULT_SETTINGS);
  setRagSettings(settings);

  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useRagThreadRuntime,
    adapter: ragThreadListAdapter,
  });

  const setModel = useCallback(
    (model: string) => setSettings((current) => ({ ...current, model })),
    [],
  );

  const settingsValue = useMemo(
    () => ({ settings, setModel }),
    [settings, setModel],
  );

  return (
    <RagSettingsProvider value={settingsValue}>
      <AssistantRuntimeProvider runtime={runtime}>
        <div className="flex h-dvh">
          <ThreadListSidebar />
          <div className="min-w-0 flex-1">
            <Thread />
          </div>
        </div>
      </AssistantRuntimeProvider>
    </RagSettingsProvider>
  );
};
