#!/bin/bash

games=games.tmp

pats=out/$RANDOM$RANDOM
patCsv=$pats.csv
patJs=$pats.js

node mine-pats-selection.js --file $games > $patCsv
node gen-patterns-data.js --file $patCsv > $patJs
PLAYOUTS=2000 PATTERN_EQUIV=50 PAT_DATA=$patJs node selfplay.js --p1 rave --p2 ravepat --size 9 --budget 1

