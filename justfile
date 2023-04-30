build-time:
  cargo build --bin okolors --profile time --features time,binary

build-bin:
  cargo build --bin okolors --features binary

clippy:
  cargo clippy --all-features
