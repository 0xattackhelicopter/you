# Stage 1: Build the Rust application
FROM rust@sha256:80729b1687999357d0ff63e1e1e68a4f2f7a4788fdf51f23442feddd18eeef41 AS builder

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

# Remove dummy main.rs and copy actual source code
RUN rm -rf src
COPY src ./src
COPY static ./static

# Build the actual application
RUN cargo build --release

# Stage 2: Create a runtime image with FFmpeg
FROM debian@sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890

# Install FFmpeg and runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/target/release/hearthly-api /usr/local/bin/hearthly-api

# Set environment variables
ENV PORT=8080
EXPOSE 8080

# Run the binary
CMD ["/usr/local/bin/hearthly-api"]