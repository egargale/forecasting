FROM runpod/pytorch:0.7.0-cu1290-torch271-ubuntu2004

WORKDIR /app

# Install minimal system dependencies including git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules https://github.com/egargale/forecasting.git /app && \
    cd forecasting

RUN pip install -r requirements.txt
RUN pip install --no-cache-dir runpod pandas

# Start the container
CMD ["python", "-u", "rp_handler.py"]