# Stage 1: Build the Rust application
FROM rust:1.86-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/app

# Copy Cargo.toml and Cargo.lock
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies to cache them
RUN cargo build --release

# Remove dummy and add actual source code
RUN rm -rf src
COPY src ./src
COPY static ./static

# Final application build
RUN cargo build --release

# Stage 2: Create a runtime image with FFmpeg
FROM debian:bookworm-slim

# Install FFmpeg and runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from builder
COPY --from=builder /usr/src/app/target/release/hearthly-api /usr/local/bin/hearthly-api

# Set environment variables and expose port
ENV PORT=8080
EXPOSE 8080

# Run the binary
CMD ["/usr/local/bin/hearthly-api"]
