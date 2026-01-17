import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";

import { Composer } from "@/components/Composer";

describe("Composer", () => {
  it("calls onSubmit with the typed message", () => {
    const handleSubmit = vi.fn();
    render(<Composer onSubmit={handleSubmit} />);

    const input = screen.getByPlaceholderText("Ask TheLaw");
    fireEvent.change(input, { target: { value: "What does the law say about fraud?" } });

    const button = screen.getByRole("button", { name: /send message/i });
    fireEvent.click(button);

    expect(handleSubmit).toHaveBeenCalledWith("What does the law say about fraud?");
  });

  it("does not submit empty values", () => {
    const handleSubmit = vi.fn();
    render(<Composer onSubmit={handleSubmit} />);

    const button = screen.getByRole("button", { name: /send message/i });
    fireEvent.click(button);

    expect(handleSubmit).not.toHaveBeenCalled();
  });
});
