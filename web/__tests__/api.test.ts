import { describe, expect, it, vi } from "vitest";

describe("api client", () => {
  it("uses NEXT_PUBLIC_API_BASE_URL when set", async () => {
    process.env.NEXT_PUBLIC_API_BASE_URL = "https://example-workers.dev";

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ answer: "ok", chunks: [], retrieval_used: false, metadata: {} }),
    });

    vi.stubGlobal("fetch", fetchMock);

    const { sendChat } = await import("@/lib/api");
    await sendChat({ message: "Hello", history: [] });

    expect(fetchMock).toHaveBeenCalledWith(
      "https://example-workers.dev/v1/chat",
      expect.objectContaining({ method: "POST" }),
    );

    vi.unstubAllGlobals();
  });
});
