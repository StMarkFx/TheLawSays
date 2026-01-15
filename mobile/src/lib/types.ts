export type Role = "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
}

export interface Chunk {
  id: number;
  source: string;
  jurisdiction: string;
  text: string;
  score?: number;
  meta?: Record<string, string>;
}

export interface ChatRequest {
  message: string;
  history: Array<{ role: Role; content: string }>;
  jurisdiction?: string | null;
  top_k?: number;
}

export interface ChatResponse {
  answer: string;
  chunks: Chunk[];
  retrieval_used: boolean;
  metadata: Record<string, unknown>;
}

export interface FeedbackPayload {
  conversation_id: string;
  message_id?: string;
  rating: "thumbs_up" | "thumbs_down";
  comment?: string;
}
