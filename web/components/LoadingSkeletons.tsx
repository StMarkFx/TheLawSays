"use client";

import { Skeleton } from "./Skeleton";

export function HeroSkeleton() {
  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col items-center gap-4 px-6 text-center">
      <Skeleton className="h-10 w-10/12 rounded-[999px]" />
      <Skeleton className="h-4 w-11/12 rounded-full" />
      <Skeleton className="h-4 w-9/12 rounded-full" />
      <Skeleton className="mt-12 h-[62px] w-full rounded-[22px]" />
    </div>
  );
}

export function MessageSkeleton({ variant = "assistant" }: { variant?: "assistant" | "user" }) {
  if (variant === "user") {
    return (
      <div className="flex w-full justify-end">
        <div className="w-full max-w-md rounded-[22px] rounded-br-sm bg-[#2D2D2D] p-5">
          <Skeleton className="h-4 w-5/6 rounded-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex w-full justify-start">
      <div className="w-full max-w-2xl space-y-4 rounded-[22px] rounded-bl-sm bg-[#1E1E1E] p-5">
        <Skeleton className="h-4 w-11/12 rounded-full" />
        <Skeleton className="h-4 w-10/12 rounded-full" />
        <Skeleton className="h-4 w-8/12 rounded-full" />
        <div className="space-y-3 border-l border-[#555555] pl-4">
          <Skeleton className="h-3 w-9/12 rounded-full" />
          <Skeleton className="h-3 w-7/12 rounded-full" />
        </div>
        <Skeleton className="h-4 w-9/12 rounded-full" />
      </div>
    </div>
  );
}

export function SourcesSkeleton() {
  return (
    <div className="space-y-3 rounded-2xl border border-[#333333] bg-[#121212] p-4">
      <div className="flex items-center justify-between rounded-xl border border-[#444444] bg-[#1E1E1E] px-4 py-3">
        <Skeleton className="h-4 w-40 rounded-full" />
        <Skeleton className="h-4 w-6 rounded-full" />
      </div>
      {[1, 2].map((item) => (
        <div key={item} className="space-y-3 rounded-2xl border border-[#333333] bg-[#0A0A0A] p-4">
          <Skeleton className="h-4 w-2/3 rounded-full" />
          <Skeleton className="h-3 w-full rounded-full" />
          <Skeleton className="h-3 w-5/6 rounded-full" />
        </div>
      ))}
    </div>
  );
}

export function SidebarSkeleton() {
  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto">
      <div className="space-y-3">
        <Skeleton className="h-4 w-3/4 rounded-full" />
        <Skeleton className="h-4 w-2/3 rounded-full" />
      </div>
      <Skeleton className="h-12 w-full rounded-[14px]" />
      <div className="space-y-3">
        <Skeleton className="h-3 w-1/2 rounded-full" />
        <Skeleton className="h-3 w-full rounded-full" />
        <Skeleton className="h-3 w-10/12 rounded-full" />
        <Skeleton className="h-3 w-8/12 rounded-full" />
      </div>
      <div className="space-y-3">
        <Skeleton className="h-3 w-1/2 rounded-full" />
        <Skeleton className="h-3 w-full rounded-full" />
        <Skeleton className="h-3 w-10/12 rounded-full" />
      </div>
      <div className="flex gap-3">
        <Skeleton className="h-10 w-full rounded-full" />
        <Skeleton className="h-10 w-full rounded-full" />
      </div>
    </div>
  );
}
