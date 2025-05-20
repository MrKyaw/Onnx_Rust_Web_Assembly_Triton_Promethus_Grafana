# Stage 1: build the Rust/WebAssembly package
FROM rust:1.83-slim AS wasm-builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh && \
    rustup target add wasm32-unknown-unknown

COPY Cargo.toml ./
COPY src ./src
RUN wasm-pack build --target web --release

# Stage 2: Python gateway
FROM python:3.12-slim
WORKDIR /app

COPY --from=wasm-builder /app/pkg /app/pkg
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY triton_gateway ./triton_gateway
COPY app ./app

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
