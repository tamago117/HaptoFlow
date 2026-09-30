FROM python:3.12-slim-bookworm

# X11 client for GUI windows
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# uv (pinned; matches the version uv.lock was produced with)
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_FROZEN=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --no-cache

RUN mkdir -m 1777 /tmp/home

CMD ["/bin/bash"]
