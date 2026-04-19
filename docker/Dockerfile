FROM nvcr.io/nvidia/nemo:24.09

WORKDIR /app

# Install only serving dependencies (training deps already in base image)
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.0" \
    pydantic==2.9.0 \
    python-dotenv==1.0.0 \
    PyYAML==6.0.2

# Copy source code (scripts/ and data/ excluded via .dockerignore)
COPY api/ ./api/
COPY pipeline/ ./pipeline/
COPY models/configs/ ./models/configs/

# Checkpoints and tokenizer are mounted as a volume at runtime
# docker run -v $(pwd)/models/checkpoints:/app/models/checkpoints ...

EXPOSE 8000

ENV TRANSFORMERS_VERBOSITY=error

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
