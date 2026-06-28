#!/bin/sh
set -e
cd "$(dirname "$0")"
CC="${CC:-cc}"
CFLAGS="${CFLAGS:--O3 -march=native -flto -funroll-loops -ffast-math -Wall -Wextra}"

$CC $CFLAGS -o test_game2.bin           game2.c test_game2.c -lm
$CC $CFLAGS -o test_game3.bin           game2.c game3.c test_game3.c -lm
$CC $CFLAGS -o test_ladder2.bin         game2.c game3.c ladder2.c test_ladder2.c -lm
$CC $CFLAGS -o test_ppat.bin            game2.c game3.c ladder2.c ppat.c test_ppat.c -lm
$CC $CFLAGS -o test_rave.bin            game2.c rave.c test_rave.c -lm
$CC $CFLAGS -o gen_evals.bin            game2.c rave.c gen_evals.c -lm
$CC $CFLAGS -o eval_compare.bin         game2.c rave.c eval_compare.c -lm
$CC $CFLAGS -o selfplay.bin             game2.c rave.c selfplay.c -lm
$CC $CFLAGS -o tune_rave.bin            game2.c rave.c tune_rave.c -lm
$CC $CFLAGS -o train_ppat.bin           game2.c game3.c ladder2.c ppat.c train_ppat.c -lm
$CC $CFLAGS -o find_ppat_features.bin   game2.c game3.c ladder2.c ppat.c find_ppat_features.c -lm

# featurepol: build without -ffast-math so softmax FP matches JS IEEE semantics.
FP_CFLAGS="-O3 -march=native -funroll-loops -Wall -Wextra"
FP_SRC="game2.c game3.c ladder2.c featurepol.c"
$CC $FP_CFLAGS -o test_featurepol.bin           $FP_SRC test_featurepol.c -lm
$CC $FP_CFLAGS -o test_fp_train.bin             $FP_SRC test_fp_train.c -lm
$CC $FP_CFLAGS -o train_featurepol_reinforce.bin $FP_SRC train_featurepol_reinforce.c -lm

echo "built 14 binaries"
