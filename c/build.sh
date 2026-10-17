#!/bin/sh
set -e
cd "$(dirname "$0")"
CC="${CC:-cc}"
CFLAGS="${CFLAGS:--O3 -march=native -flto -funroll-loops -ffast-math -Wall -Wextra}"

$CC $CFLAGS -o test_game2.bin           game2.c test_game2.c -lm
$CC $CFLAGS -o test_game3.bin           game2.c game3.c test_game3.c -lm
$CC $CFLAGS -o test_ladder2.bin         game2.c game3.c ladder2.c test_ladder2.c -lm
$CC $CFLAGS -o test_ppat.bin            game2.c ppat.c test_ppat.c -lm
$CC $CFLAGS -o bench_ppat.bin           game2.c ppat.c bench_ppat.c -lm
$CC $CFLAGS -o test_rave.bin            game2.c rave.c test_rave.c -lm
$CC $CFLAGS -o gen_evals.bin            game2.c rave.c gen_evals.c -lm
$CC $CFLAGS -o eval_compare.bin         game2.c rave.c eval_compare.c -lm
$CC $CFLAGS -o selfplay.bin             game2.c rave.c selfplay.c -lm
$CC $CFLAGS -o tune_rave.bin            game2.c rave.c tune_rave.c -lm
$CC $CFLAGS -o train-ppat              game2.c ppat.c fpol.c puct.c match.c train_ppat.c -lm
$CC $CFLAGS -o find_ppat_features.bin   game2.c ppat.c find_ppat_features.c -lm
$CC $CFLAGS -o test_vpat.bin            game2.c vpat.c test_vpat.c -lm
$CC $CFLAGS -o test_fpol.bin            game2.c fpol.c test_fpol.c -lm
$CC $CFLAGS -o test_puct.bin            game2.c fpol.c ppat.c puct.c test_puct.c -lm
$CC $CFLAGS -o ppat_match.bin          game2.c fpol.c ppat.c puct.c match.c ppat_match.c -lm
$CC $CFLAGS -o test_match.bin          game2.c fpol.c ppat.c puct.c test_match.c -lm
$CC $CFLAGS -o test_cnn.bin            game2.c game3.c ladder2.c cnn.c test_cnn.c -lm
# The double-precision gradient check: the network recompiled with float = double.
$CC $CFLAGS -c -o game2-for-double.o   game2.c
$CC $CFLAGS -c -o game3-for-double.o   game3.c
$CC $CFLAGS -c -o ladder2-for-double.o ladder2.c
$CC $CFLAGS -Dfloat=double -Dexpf=exp -DCNN_TEST_DOUBLE -o test_cnn_double.bin game2-for-double.o game3-for-double.o ladder2-for-double.o cnn.c test_cnn.c -lm
rm -f game2-for-double.o game3-for-double.o ladder2-for-double.o
$CC $CFLAGS -fopenmp -o train_vnet_supervised.bin game2.c game3.c ladder2.c cnn.c fpol.c vnet_tool.c train_vnet_supervised.c -lm
$CC $CFLAGS -fopenmp -o train_vnet_td.bin game2.c game3.c ladder2.c cnn.c vnet_tool.c fpol.c ppat.c puct.c train_vnet_td.c -lm

# EXPERIMENT — symmetric 3x3 kernels (c/cnn_sym.c, a drop-in for cnn.c): the
# unchanged trainer and vnet_tool built against it, and its tests.  Delete
# this block with cnn_sym.c / test_cnn_sym.c to drop the experiment.
$CC $CFLAGS -o test_cnn_sym.bin        game2.c game3.c ladder2.c cnn_sym.c test_cnn_sym.c -lm
$CC $CFLAGS -c -o game2-for-sym-double.o   game2.c
$CC $CFLAGS -c -o game3-for-sym-double.o   game3.c
$CC $CFLAGS -c -o ladder2-for-sym-double.o ladder2.c
$CC $CFLAGS -Dfloat=double -Dexpf=exp -DCNN_TEST_DOUBLE -o test_cnn_sym_double.bin game2-for-sym-double.o game3-for-sym-double.o ladder2-for-sym-double.o cnn_sym.c test_cnn_sym.c -lm
rm -f game2-for-sym-double.o game3-for-sym-double.o ladder2-for-sym-double.o
$CC $CFLAGS -fopenmp -o train_vnet_sym.bin game2.c game3.c ladder2.c cnn_sym.c fpol.c vnet_tool.c train_vnet_supervised.c -lm
# ... and with the corner-L-shape 4-hot (-DVSYM_CORNER: 53 planes).
$CC $CFLAGS -DVSYM_CORNER -o test_cnn_sym_corner.bin game2.c game3.c ladder2.c cnn_sym.c test_cnn_sym.c -lm
$CC $CFLAGS -DVSYM_CORNER -fopenmp -o train_vnet_sym_corner.bin game2.c game3.c ladder2.c cnn_sym.c fpol.c vnet_tool.c train_vnet_supervised.c -lm
# ... and with the 3x3's edges too (-DVSYM_CORNER -DVSYM_EDGE: 71 planes).
$CC $CFLAGS -DVSYM_CORNER -DVSYM_EDGE -o test_cnn_sym_corner_edge.bin game2.c game3.c ladder2.c cnn_sym.c test_cnn_sym.c -lm
$CC $CFLAGS -DVSYM_CORNER -DVSYM_EDGE -fopenmp -o train_vnet_sym_corner_edge.bin game2.c game3.c ladder2.c cnn_sym.c fpol.c vnet_tool.c train_vnet_supervised.c -lm

echo "built 28 binaries"
