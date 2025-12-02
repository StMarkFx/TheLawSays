"""Shim entrypoint so `uvicorn main:app` keeps working after moving the API package."""

from api.main import app  # noqa: F401  (re-export for Procfile/hosts that still reference `main:app`)


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
