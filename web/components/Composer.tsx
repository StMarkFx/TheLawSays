"use client";

import { SendHorizontal } from "lucide-react";
import { FormEvent, useState } from "react";

interface ComposerProps {
  onSubmit: (message: string) => void;
  disabled?: boolean;
  className?: string;
}

export function Composer({ onSubmit, disabled, className }: ComposerProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!value.trim() || disabled) {
      return;
    }
    onSubmit(value.trim());
    setValue("");
  };

  const canSend = value.trim().length > 0 && !disabled;

  return (
    <form
      onSubmit={handleSubmit}
      className={`flex items-center gap-3 rounded-[22px] border border-border bg-input/80 px-5 py-3 backdrop-blur ${className ?? ""}`}
    >
      <input
        className="flex-1 bg-transparent text-base text-foreground placeholder:text-[#6d6d6d] focus:outline-none"
        placeholder="Ask TheLaw"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        disabled={disabled}
      />
      <button
        type="submit"
        disabled={!canSend}
        className={`flex h-11 w-11 items-center justify-center rounded-full border transition ${
          canSend
            ? "border-accent text-accent hover:bg-accent/10"
            : "border-border text-muted/70"
        } disabled:cursor-not-allowed`}
        aria-label="Send message"
      >
        <SendHorizontal size={20} />
      </button>
    </form>
  );
}
