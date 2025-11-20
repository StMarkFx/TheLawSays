import type { ChatRequest, ChatResponse, FeedbackPayload } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const error = (detail && detail.detail) || response.statusText;
    throw new Error(typeof error === "string" ? error : "Unexpected error from API");
  }
  return response.json() as Promise<T>;
}

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse<ChatResponse>(response);
}

export async function sendFeedback(payload: FeedbackPayload): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await handleResponse(response);
}
