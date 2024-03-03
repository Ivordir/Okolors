check:
  cargo fmt --check
  typos
  cargo doc --no-deps --lib
  cargo hack --feature-powerset --skip default,_internal clippy -- -D warnings

test:
  cargo test

test-hack:
  cargo test --doc
  cargo hack -p okolors --feature-powerset --skip default,_internal test --lib
  cargo hack -p okolors-bin --feature-powerset --skip default test
