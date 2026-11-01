FROM python:3.14.4-slim

WORKDIR /src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# create a non-root user
RUN useradd -m appuser

# copy app files with correct ownership
COPY --chown=appuser:appuser . .

# run as non-root
USER appuser

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000}"]