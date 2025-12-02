"use client";

import { Clipboard, Loader2 } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import { useEffect, useMemo, useRef, useState } from "react";

import type { Chunk, Message } from "@/lib/types";

import { CitationAccordion } from "./CitationAccordion";
import { FeedbackButtons } from "./FeedbackButtons";
import { HeroSkeleton } from "./LoadingSkeletons";

interface ChatWindowProps {
  messages: Message[];
  loading: boolean;
  chunks: Chunk[];
  retrievalUsed: boolean;
  onFeedback: (rating: "thumbs_up" | "thumbs_down") => void;
  onCopyAnswer: (content: string) => void;
}

const markdownComponents: Components = {
  blockquote: ({ children }) => (
    <blockquote className="border-l border-quote pl-4 text-sm text-[#a0a0a0]">{children}</blockquote>
  ),
  p: ({ children }) => <p className="text-base leading-relaxed text-foreground">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
};

export function ChatWindow({ messages, loading, chunks, retrievalUsed, onFeedback, onCopyAnswer }: ChatWindowProps) {
  const [feedbackGiven, setFeedbackGiven] = useState(false);
  const scrollRef = useRef<HTMLElement>(null);

  const assistantMessages = useMemo(() => messages.filter((message) => message.role === "assistant"), [messages]);
  const latestAssistant = assistantMessages[assistantMessages.length - 1];
  const showEmptyState = messages.length === 0;

  useEffect(() => {
    setFeedbackGiven(false);
  }, [latestAssistant?.id]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleFeedback = (rating: "thumbs_up" | "thumbs_down") => {
    onFeedback(rating);
    setFeedbackGiven(true);
  };

  return (
    <section
      ref={scrollRef}
      className="mx-auto flex w-full max-w-3xl flex-1 flex-col overflow-y-auto px-4 pt-6 pb-6 md:px-0"
    >
      {showEmptyState ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center text-muted">
          {loading ? (
            <HeroSkeleton />
          ) : (
            <>
              <p className="text-2xl font-medium text-foreground">
                Empower yourself with Nigerian law—at your fingertips.
              </p>
              <p className="text-base">
                Ask any question about Federal or Lagos State statutes and get instant, verifiable answers
                backed by exact sections and citations.
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="flex flex-1 flex-col gap-6">
          {messages.map((message) => {
            const isUser = message.role === "user";
            const isLatestAssistant = !isUser && message.id === latestAssistant?.id;

            if (isUser) {
              return (
                <div key={message.id} className="flex w-full justify-end">
                  <div className="inline-flex max-w-[80%] rounded-[22px] bg-bubbleUser px-5 py-3 text-base text-white">
                    <p className="whitespace-pre-line leading-relaxed">{message.content}</p>
                  </div>
                </div>
              );
            }

            return (
              <div key={message.id} className="flex w-full justify-start">
                <div className="flex w-full max-w-[80%] flex-col gap-4">
                  <article className="space-y-4 text-base leading-relaxed text-foreground">
                    <ReactMarkdown components={markdownComponents}>{message.content}</ReactMarkdown>
                  </article>

                  <div className="flex items-center gap-3 text-sm text-muted">
                    <button
                      type="button"
                      onClick={() => onCopyAnswer(message.content)}
                      className="flex h-9 w-9 items-center justify-center rounded-full border border-border text-muted transition hover:border-accent hover:text-accent"
                      aria-label="Copy answer"
                    >
                      <Clipboard size={16} />
                    </button>
                    <FeedbackButtons onFeedback={handleFeedback} disabled={feedbackGiven} />
                  </div>

                  {isLatestAssistant && retrievalUsed && chunks.length > 0 && (
                    <CitationAccordion chunks={chunks} className="mt-2" />
                  )}
                </div>
              </div>
            );
          })}

          {loading && (
            <div className="flex items-center gap-3 text-sm text-muted">
              <Loader2 size={16} className="animate-spin text-accent" />
              Thinking...
            </div>
          )}
        </div>
      )}
    </section>
  );
}
