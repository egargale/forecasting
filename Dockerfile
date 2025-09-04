# Use RunPod CUDA-enabled base image to resolve CUDA_HOME issues
FROM runpod/base:0.7.0-cuda1290-ubuntu2404

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda

# Change the working directory to the `app` directory
WORKDIR /app

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

# Install additional requirements
RUN uv pip install -r requirements.txt

# Start the container
CMD ["uv", "run", "rp_handler.py"]