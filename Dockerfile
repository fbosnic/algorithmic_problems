FROM rust:1.71.1-slim
RUN apt-get update && apt-get install -y build-essential gdb
