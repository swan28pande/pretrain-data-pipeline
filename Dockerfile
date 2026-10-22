# Pre-training data pipeline: local engine and PySpark (local[*] or as a Spark client).
# Debian bookworm base: ships OpenJDK 17 (Java 17+ is required by PySpark 4).
FROM python:3.11-slim-bookworm

# Java runtime for PySpark. JAVA_HOME is symlinked so it works on amd64 and arm64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps zip \
    && ln -s "$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" /opt/java \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/java \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON=python \
    PYSPARK_PYTHON=python

WORKDIR /app
COPY requirements.txt .
# Compilers are only needed where fasttext has no prebuilt wheel; removed in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY config.yaml .
COPY src/ src/
COPY spark/ spark/
COPY scripts/ scripts/

# Pipeline outputs and downloaded models; mount volumes here to keep them.
VOLUME ["/app/work", "/app/models"]

ENTRYPOINT ["scripts/run_pipeline.sh"]
CMD ["local"]
