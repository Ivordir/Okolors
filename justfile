build-time:
  cargo build --bin okolors --features time,binary --profile time

build-bin:
  cargo build --bin okolors --features binary --release

clippy:
  cargo clippy --all-features

test:
  cargo test --features binary
