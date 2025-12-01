"use client";

import { useCallback, useMemo, useState } from "react";
import { v4 as uuid } from "uuid";

import { ChatWindow } from "@/components/ChatWindow";
import { Composer } from "@/components/Composer";
import { LayoutShell } from "@/components/LayoutShell";
import { Sidebar } from "@/components/Sidebar";
import { sendChat, sendFeedback } from "@/lib/api";
import type { Chunk, Message } from "@/lib/types";

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [retrievalUsed, setRetrievalUsed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [conversationId, setConversationId] = useState<string>(() => uuid());
  const [error, setError] = useState<string | null>(null);

  const historyPayload = useMemo(
    () => messages.map(({ role, content }) => ({ role, content })),
    [messages],
  );

  const handleSubmit = useCallback(
    async (content: string) => {
      if (loading) return;
      setError(null);
      const userMessage: Message = {
        id: uuid(),
        role: "user",
        content,
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);
      const nextHistory = [...historyPayload, { role: "user", content }];
      setLoading(true);
      try {
        const response = await sendChat({
          message: content,
          history: nextHistory,
        });
        const assistantMessage: Message = {
          id: uuid(),
          role: "assistant",
          content: response.answer,
          createdAt: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setChunks(response.chunks);
        setRetrievalUsed(response.retrieval_used);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      } finally {
        setLoading(false);
      }
    },
    [historyPayload, loading],
  );

  const handleFeedback = useCallback(
    async (rating: "thumbs_up" | "thumbs_down") => {
      if (!conversationId) return;
      try {
        await sendFeedback({
          conversation_id: conversationId,
          rating,
        });
      } catch (err) {
        console.error("Failed to send feedback", err);
      }
    },
    [conversationId],
  );

  const handleCopyAnswer = useCallback((content: string) => {
    if (navigator?.clipboard) {
      navigator.clipboard.writeText(content).catch(() => {
        setError("Unable to copy answer to clipboard.");
      });
    }
  }, []);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setChunks([]);
    setRetrievalUsed(false);
    setConversationId(uuid());
    setError(null);
  }, []);

  return (
    <LayoutShell
      onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
      onNewChat={handleNewChat}
    >
      <Sidebar open={isSidebarOpen} onClose={() => setSidebarOpen(false)} loading={loading} />

      {isSidebarOpen && (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/70"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close sidebar overlay"
        />
      )}

      <div className="flex flex-1 justify-center px-4 pb-40 pt-6 md:px-8">
        <ChatWindow
          messages={messages}
          loading={loading}
          chunks={retrievalUsed ? chunks : []}
          retrievalUsed={retrievalUsed}
          onFeedback={handleFeedback}
          onCopyAnswer={handleCopyAnswer}
        />
      </div>

      <div className="fixed bottom-0 left-0 right-0 z-40 px-4 pb-5 pt-2 md:px-8">
        <div className="mx-auto flex w-full max-w-3xl flex-col items-center gap-2">
          {error && <p className="w-full text-center text-sm text-red-400">{error}</p>}
          <Composer onSubmit={handleSubmit} disabled={loading} className="w-full" />
          <p className="text-center text-xs text-muted">
            Built for educational research. Consult qualified lawyers for legal advice.
          </p>
        </div>
      </div>
    </LayoutShell>
  );
}
