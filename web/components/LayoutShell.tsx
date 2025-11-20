"use client";

import { Menu, Plus } from "lucide-react";
import { ReactNode } from "react";

interface LayoutShellProps {
  onToggleSidebar: () => void;
  onNewChat: () => void;
  children: ReactNode;
}

export function LayoutShell({ onToggleSidebar, onNewChat, children }: LayoutShellProps) {
  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="fixed inset-x-0 top-0 z-40 flex h-16 items-center justify-between gap-4 border-b border-border bg-background/95 px-4 backdrop-blur md:px-8">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="flex h-11 w-11 items-center justify-center rounded-full border border-border bg-surface text-foreground transition hover:border-accent hover:text-accent"
          aria-label="Toggle sidebar"
        >
          <Menu size={22} />
        </button>
        <div className="flex flex-1 items-center justify-center">
          <div className="flex items-center gap-2 rounded-[30px] border border-border bg-card px-5 py-2 text-sm font-semibold">
            <span role="img" aria-label="justice scale">⚖️</span>
            <span>The Law Says</span>
          </div>
        </div>
        <button
          type="button"
          onClick={onNewChat}
          className="flex h-11 w-11 items-center justify-center rounded-full border border-border bg-surface text-foreground transition hover:border-accent hover:text-accent"
          aria-label="Start new chat"
        >
          <Plus size={22} />
        </button>
      </header>
      <main className="flex flex-1 flex-col pt-16">{children}</main>
    </div>
  );
}
