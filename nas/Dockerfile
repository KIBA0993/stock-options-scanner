FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

# Cron for containers (no daemon required)
ARG SUPERCRONIC_VERSION=v0.2.33
ARG TARGETARCH=amd64
RUN curl -fsSL "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${TARGETARCH}" \
        -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY nas/scripts/ /scripts/
COPY nas/crontab /etc/crontab

RUN chmod +x /scripts/*.sh

# scanner.py uses Path.home() / "trading"
ENV HOME=/data
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York

ENTRYPOINT ["/scripts/entrypoint.sh"]
