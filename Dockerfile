FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_viva.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements_viva.txt

COPY . .

EXPOSE 7860

CMD ["python", "viva_server.py"]
