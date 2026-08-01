import {
  useLocalRuntime,
  type RemoteThreadListAdapter,
} from "@assistant-ui/react";
import { createAssistantStream } from "assistant-stream";
import { createRagAdapter } from "@/lib/rag-adapter";
import { DEFAULT_SETTINGS, type RagSettings } from "@/lib/rag";

/**
 * useRemoteThreadListRuntime re-registers its runtime hook whenever the hook's
 * identity changes, so the hook must be stable and cannot close over React
 * state. Settings therefore live in a module holder that the UI writes to.
 * This module is only ever evaluated in the browser ("use client" callers), so
 * the holder is per-tab, not shared across server requests.
 */
const settingsHolder = { current: DEFAULT_SETTINGS };

export const setRagSettings = (settings: RagSettings) => {
  settingsHolder.current = settings;
};

const ragChatAdapter = createRagAdapter(() => settingsHolder.current);

/** Per-thread runtime. One instance is kept alive per open thread. */
export const useRagThreadRuntime = () => useLocalRuntime(ragChatAdapter);

type StoredThread = {
  remoteId: string;
  status: "regular" | "archived";
  title?: string | undefined;
  lastMessageAt: Date;
};

const TITLE_MAX = 60;

/**
 * Client-side thread registry. Threads and their messages live in memory for
 * the life of the tab; persisting them across reloads would need a thread and
 * message store behind the FastAPI service.
 */
const threads = new Map<string, StoredThread>();

export const ragThreadListAdapter: RemoteThreadListAdapter = {
  async list() {
    return {
      threads: [...threads.values()]
        .sort((a, b) => b.lastMessageAt.getTime() - a.lastMessageAt.getTime())
        .map((thread) => ({
          remoteId: thread.remoteId,
          status: thread.status,
          title: thread.title,
          lastMessageAt: thread.lastMessageAt,
        })),
    };
  },

  async initialize(threadId) {
    threads.set(threadId, {
      remoteId: threadId,
      status: "regular",
      lastMessageAt: new Date(),
    });
    return { remoteId: threadId, externalId: undefined };
  },

  async rename(remoteId, title) {
    const thread = threads.get(remoteId);
    if (thread) thread.title = title;
  },

  async archive(remoteId) {
    const thread = threads.get(remoteId);
    if (thread) thread.status = "archived";
  },

  async unarchive(remoteId) {
    const thread = threads.get(remoteId);
    if (thread) thread.status = "regular";
  },

  async delete(remoteId) {
    threads.delete(remoteId);
  },

  async fetch(remoteId) {
    const thread = threads.get(remoteId);
    if (!thread) throw new Error(`Thread "${remoteId}" not found`);
    return {
      remoteId: thread.remoteId,
      status: thread.status,
      title: thread.title,
      lastMessageAt: thread.lastMessageAt,
    };
  },

  // Titles come from the first question rather than a second LLM call, which
  // would double the Bedrock cost of every new thread.
  async generateTitle(remoteId, messages) {
    const firstQuestion = messages
      .find((message) => message.role === "user")
      ?.content.map((part) => (part.type === "text" ? part.text : ""))
      .join("")
      .trim();

    const title = !firstQuestion
      ? "New Chat"
      : firstQuestion.length > TITLE_MAX
        ? `${firstQuestion.slice(0, TITLE_MAX).trimEnd()}…`
        : firstQuestion;

    const thread = threads.get(remoteId);
    if (thread) {
      thread.title = title;
      thread.lastMessageAt = new Date();
    }

    return createAssistantStream(async (controller) => {
      controller.appendText(title);
    });
  },
};
