build-time:
  cargo build --bin okolors --features time --profile bench

bench-func func:
  cargo bench --bench functions -- {{func}}

experiment-param param:
  cargo run --release --example parameters -- {{param}}
