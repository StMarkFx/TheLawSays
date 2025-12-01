"use client";

import { Github, Linkedin, Mail, X } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";

import { SidebarSkeleton } from "./LoadingSkeletons";

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  loading?: boolean;
}

export function Sidebar({ open, onClose, loading }: SidebarProps) {
  const classes = useMemo(
    () =>
      [
        "fixed inset-y-0 left-0 z-50 w-[85%] max-w-sm transform bg-card text-foreground shadow-2xl transition-transform duration-200",
        open ? "translate-x-0" : "-translate-x-full",
      ].join(" "),
    [open],
  );

  return (
    <aside className={classes}>
      <div className="flex h-full flex-col">
        <div className="relative flex-1 overflow-hidden">
          {loading && (
            <div className="pointer-events-none absolute inset-0 z-10 bg-background/95 px-6 py-6">
              <SidebarSkeleton />
            </div>
          )}

          <div className="flex h-full flex-col gap-6 overflow-y-auto px-6 py-6">
            <section>
              <h3 className="text-xs uppercase tracking-[0.2em] text-muted">Project Info</h3>
              <p className="mt-3 text-sm leading-relaxed text-foreground">
                TheLawSays is an open-source AI legal assistant built to democratize access to Nigerian law. It uses advanced
                retrieval-augmented generation to deliver precise, citation-backed answers from official Federal and Lagos State
                statutes in seconds. Informative, not legal advice.
              </p>
              <Link
                href="https://github.com/StMarkFx/TheLawSays"
                target="_blank"
                className="mt-3 inline-flex items-center gap-2 text-sm text-foreground underline decoration-dotted underline-offset-4 transition hover:text-accent"
              >
                <Github size={16} />
                GitHub: github.com/StMarkFx/TheLawSays
              </Link>
            </section>

            <section className="space-y-3 text-sm leading-relaxed text-foreground">
              <h3 className="text-xs uppercase tracking-[0.2em] text-muted">About St. Mark</h3>
              <p>
                St. Mark Adebayo is an AI/ML Engineer who created TheLawSays, a Nigerian legal AI assistant designed to democratize
                access to legal information through advanced RAG technology.
              </p>
              <p>
                He specializes in applying machine learning and artificial intelligence to solve real-world problems. Beyond code,
                St. Mark is passionate about how data, AI, and human insight can come together to address challenges across justice,
                education, innovation, and community growth.
              </p>
              <div className="space-y-2 text-sm">
                <Link
                  href="https://linkedin.com/in/stmarkadebayo"
                  target="_blank"
                  className="flex items-center gap-2 text-foreground underline decoration-dotted underline-offset-4 transition hover:text-accent"
                >
                  <Linkedin size={16} />
                  linkedin.com/in/stmarkadebayo
                </Link>
                <Link
                  href="https://github.com/StMarkFx"
                  target="_blank"
                  className="flex items-center gap-2 text-foreground underline decoration-dotted underline-offset-4 transition hover:text-accent"
                >
                  <Github size={16} />
                  github.com/StMarkFx
                </Link>
                <Link
                  href="mailto:stmarkadebayo@gmail.com"
                  className="flex items-center gap-2 text-foreground underline decoration-dotted underline-offset-4 transition hover:text-accent"
                >
                  <Mail size={16} />
                  stmarkadebayo@gmail.com
                </Link>
              </div>
            </section>
          </div>
        </div>
      </div>
    </aside>
  );
}
