"use client";

import { createContext, useContext } from "react";
import type { RagSettings } from "@/lib/rag";

type RagSettingsContextValue = {
  settings: RagSettings;
  setModel: (model: string) => void;
};

const RagSettingsContext = createContext<RagSettingsContextValue | null>(null);

export const RagSettingsProvider = RagSettingsContext.Provider;

export const useRagSettings = () => {
  const value = useContext(RagSettingsContext);
  if (!value) {
    throw new Error("useRagSettings must be used inside <RagSettingsProvider>");
  }
  return value;
};
