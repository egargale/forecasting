# FROM runpod/pytorch:0.7.0-cu1290-torch271-ubuntu2004
# Install uv
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app

# Install minimal system dependencies including git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy the project into the image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# RUN git clone --recurse-submodules https://github.com/egargale/forecasting.git /app 

RUN uv pip install -r requirements.txt
# RUN pip install --no-cache-dir runpod pandas

# Start the container
# CMD ["python", "-u", "rp_handler.py"]
CMD ["uv", "run", "rp_handler.py"]