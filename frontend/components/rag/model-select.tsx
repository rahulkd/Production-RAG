"use client";

import { ChevronDownIcon, CpuIcon } from "lucide-react";
import type { FC } from "react";
import { useRagSettings } from "@/components/rag/rag-settings";
import { RAG_MODELS, modelLabel } from "@/lib/rag";

export const ModelSelect: FC = () => {
  const { settings, setModel } = useRagSettings();

  return (
    <label className="border-border/60 text-muted-foreground hover:bg-muted-foreground/10 dark:border-muted-foreground/15 relative flex h-7 cursor-pointer items-center gap-1.5 rounded-full border ps-2.5 pe-1.5 text-xs font-medium transition-colors">
      <CpuIcon className="size-3.5 shrink-0" aria-hidden />
      <span className="text-foreground whitespace-nowrap">
        {modelLabel(settings.model)}
      </span>
      <ChevronDownIcon className="size-3.5 shrink-0" aria-hidden />
      <select
        aria-label="Generation model"
        value={settings.model}
        onChange={(event) => setModel(event.target.value)}
        className="absolute inset-0 cursor-pointer opacity-0"
      >
        {RAG_MODELS.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label} — {model.provider}
          </option>
        ))}
      </select>
    </label>
  );
};
