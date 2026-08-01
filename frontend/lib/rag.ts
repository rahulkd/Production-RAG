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
    id: "meta.llama3-70b-instruct-v1:0",
    label: "Llama 3 70B Instruct",
    provider: "AWS Bedrock",
  },
];

export const DEFAULT_MODEL_ID = RAG_MODELS[0]!.id;

/** Backend caps top_k at 10 (AskRequest: ge=1, le=10). */
export const DEFAULT_TOP_K = 3;
export const DEFAULT_USE_HYBRID = true;

export type RagSettings = {
  model: string;
  topK: number;
  useHybrid: boolean;
  categories?: readonly string[] | undefined;
};

export const DEFAULT_SETTINGS: RagSettings = {
  model: DEFAULT_MODEL_ID,
  topK: DEFAULT_TOP_K,
  useHybrid: DEFAULT_USE_HYBRID,
};

export const modelLabel = (id: string) =>
  RAG_MODELS.find((model) => model.id === id)?.label ?? id;

const ARXIV_ID = /(\d{4}\.\d{4,5})(v\d+)?/;

/** Display label for a source PDF URL, e.g. "arXiv:2607.15267". */
export const sourceTitle = (url: string) => {
  const arxivId = ARXIV_ID.exec(url)?.[1];
  return arxivId ? `arXiv:${arxivId}` : url;
};
