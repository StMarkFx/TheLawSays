"use client";

import { ThumbsDown, ThumbsUp } from "lucide-react";

interface FeedbackButtonsProps {
  onFeedback: (rating: "thumbs_up" | "thumbs_down") => void;
  disabled?: boolean;
}

export function FeedbackButtons({ onFeedback, disabled }: FeedbackButtonsProps) {
  const baseClasses =
    "flex h-9 w-9 items-center justify-center rounded-full border border-border text-muted transition hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-60";

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => onFeedback("thumbs_up")}
        disabled={disabled}
        className={baseClasses}
        aria-label="Mark answer helpful"
      >
        <ThumbsUp size={16} />
      </button>
      <button
        type="button"
        onClick={() => onFeedback("thumbs_down")}
        disabled={disabled}
        className={baseClasses}
        aria-label="Mark answer not helpful"
      >
        <ThumbsDown size={16} />
      </button>
    </div>
  );
}
