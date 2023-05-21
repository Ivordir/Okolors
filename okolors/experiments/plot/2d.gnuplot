#! /usr/bin/gnuplot

plot \
  file index "Colors" using "a":"b":"n":"color" with points ps variable pt 5 lc rgbcolor variable, \
  file index "Centroids" using "a":"b":"n":"color" with points ps variable pt 3 lc rgbcolor variable lw 5

pause mouse close
