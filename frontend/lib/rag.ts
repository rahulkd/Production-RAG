export type RagModel = {
  readonly id: string;
  readonly label: string;
  readonly provider: string;
};

/**
 * Mirrors the `model` field accepted by the backend's AskRequest. Adding a model
 * here is not enough on its own — it must also be enabled in AWS Bedrock for the
 * account the API authenticates as.
 */
export const RAG_MODELS: readonly RagModel[] = [
  {
    id: "us.meta.llama3-3-70b-instruct-v1:0",
    label: "Llama 3.3 70B Instruct",
    provider: "AWS Bedrock",
  },
];

export const DEFAULT_MODEL_ID = RAG_MODELS[0]!.id;

/** Backend caps top_k at 10 (AskRequest: ge=1, le=10). */
export const DEFAULT_TOP_K = 10;
export const DEFAULT_USE_HYBRID = true;

/** Backend endpoint a question is sent to: /api/v1/stream or /api/v1/ask. */
export type RagEndpoint = "stream" | "ask";

export type RagSettings = {
  model: string;
  topK: number;
  useHybrid: boolean;
  endpoint: RagEndpoint;
  categories?: readonly string[] | undefined;
};

export const DEFAULT_SETTINGS: RagSettings = {
  model: DEFAULT_MODEL_ID,
  topK: DEFAULT_TOP_K,
  useHybrid: DEFAULT_USE_HYBRID,
  endpoint: "stream",
};

/**
 * Browser-measured latency of one question, stored on the assistant message as
 * `metadata.custom.latency`. Measured from just before the request is sent, so
 * it includes the Next.js proxy hop — what the user actually waits.
 */
export type RagLatency = {
  endpoint: RagEndpoint;
  /** Time until the first answer token arrived. Only set for /stream. */
  firstTokenMs?: number | undefined;
  totalMs: number;
};

export const formatMs = (ms: number) =>
  ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;

export const modelLabel = (id: string) =>
  RAG_MODELS.find((model) => model.id === id)?.label ?? id;

const ARXIV_ID = /(\d{4}\.\d{4,5})(v\d+)?/;

/** Display label for a source PDF URL, e.g. "arXiv:2607.15267". */
export const sourceTitle = (url: string) => {
  const arxivId = ARXIV_ID.exec(url)?.[1];
  return arxivId ? `arXiv:${arxivId}` : url;
};
