"use client";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  const classes = [
    "animate-pulse bg-gradient-to-r from-skeleton/70 via-skeletonHighlight/70 to-skeleton/70",
    "rounded-xl",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return <div className={classes} aria-hidden="true" />;
}
