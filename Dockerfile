FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_viva.txt /tmp/requirements_viva.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements_viva.txt

# Hugging Face Spaces run the container as user ID 1000, so the app
# directory must be owned by that user for runtime writes to succeed.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "viva_server.py"]
