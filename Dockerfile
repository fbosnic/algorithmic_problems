FROM rust:1.89.0-slim
RUN apt-get update && apt-get install -y build-essential gdb

