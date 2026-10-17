FROM python:3.12-slim

WORKDIR /app

# Create a non-root user for Hugging Face security compliance
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy dependency configuration and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application workspace content
COPY --chown=user . .

EXPOSE 7860

# Launch Uvicorn server pointing to FastAPI setup
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
