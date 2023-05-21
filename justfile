build-time:
  cargo build --bin okolors --features time --profile bench

bench-func func:
  cargo bench --bench functions -- {{func}}

experiment-param param:
  cargo run --release --example parameters -- {{param}}

_experiment-plot plot image *args:
  #! /usr/bin/env bash
  set -e
  image="$(realpath "{{image}}")"
  cd '{{justfile_directory()}}/okolors/experiments/plot'
  mkdir -p data
  data='data/{{file_stem(image)}}.dat'
  cargo run --release --example plot -- "$image" {{args}} > "$data"
  gnuplot -e "file='$data'" {{plot}}.gnuplot

experiment-plot-2d image *args: (_experiment-plot '2d' image args)

experiment-plot-3d image *args: (_experiment-plot '3d' image args)
