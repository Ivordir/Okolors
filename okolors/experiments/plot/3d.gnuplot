#! /usr/bin/gnuplot

splot \
  file index "Colors" using "a":"b":"l":"n":"color" with points ps variable pt 5 lc rgbcolor variable, \
  file index "Centroids" using "a":"b":"l":"n":"color" with points ps variable pt 3 lc rgbcolor variable lw 5

pause mouse close
