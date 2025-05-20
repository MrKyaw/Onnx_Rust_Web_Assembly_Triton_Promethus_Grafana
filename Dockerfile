
# Stage 1: Build WASM artifacts
FROM rust:latest AS wasm-builder

WORKDIR /app

# Install wasm tools
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh && \
    rustup target add wasm32-unknown-unknown

# First copy only Cargo.toml to generate lock file
COPY Cargo.toml .

# Create dummy source files
RUN mkdir -p src && \
    echo "use wasm_bindgen::prelude::*;" > src/lib.rs && \
    echo "#[wasm_bindgen] pub fn dummy() {}" >> src/lib.rs

# Generate fresh lock file
RUN cargo generate-lockfile

# Now copy real source files
COPY src ./src

# Verify files exist
RUN ls -la src/ && [ -f src/lib.rs ] || (echo "Error: src/lib.rs missing!" && exit 1)

# Build WASM package
RUN wasm-pack build --target web --release

# Stage 2: Build FastAPI app
FROM python:3.9-slim
WORKDIR /app
COPY --from=wasm-builder /app/pkg /app/pkg
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]