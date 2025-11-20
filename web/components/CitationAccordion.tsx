"use client";

import clsx from "clsx";
import { ChevronDown } from "lucide-react";
import { useState } from "react";

import type { Chunk } from "@/lib/types";

interface CitationAccordionProps {
  chunks: Chunk[];
  className?: string;
}

export function CitationAccordion({ chunks, className }: CitationAccordionProps) {
  const [open, setOpen] = useState(false);

  if (!chunks.length) {
    return null;
  }

  return (
    <div className={clsx("w-full rounded-2xl border border-border/80 bg-surface p-4", className)}>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center justify-between rounded-xl border border-border bg-[#141414] px-4 py-3 text-sm font-semibold text-foreground"
      >
        <span>Show retrieved sources</span>
        <ChevronDown size={18} className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="mt-4 space-y-3 text-sm text-muted">
          {chunks.map((chunk, index) => (
            <div key={`${chunk.source}-${index}`} className="space-y-2 rounded-2xl border border-border bg-[#1a1a1a] p-4">
              <p className="text-sm font-semibold text-foreground">
                Excerpt {index + 1} - {chunk.source} ({chunk.jurisdiction})
              </p>
              <p className="whitespace-pre-wrap text-sm text-muted">{chunk.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
