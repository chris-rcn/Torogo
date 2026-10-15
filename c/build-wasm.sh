#!/bin/sh
# Build the browser WebAssembly module for ppat playouts (c/wasm_ppat.c) with
# wasi-sdk:  c/build-wasm.sh  (WASI_SDK defaults to ~/opt/wasi-sdk-25.0-x86_64-linux)
#
#   ../ppat-playout.wasm       the module ppat-wasm.js loads (stripped; fatal errors via
#                              the JS hook in c/fatal.h, so no stdio)
#   ppat-playout-test.wasm     the same plus wasm_test_bench (-DWASM_PPAT_TEST)
#   bench_ppat.wasm            bench_ppat.c as a WASI program (file loader), for
#                              the hand-off check and timings under node:wasi
#   ../vnet-eval.wasm          the value net (c/wasm_vnet.c) vnet-wasm.js loads
#
# wasi-sdk 25: https://github.com/WebAssembly/wasi-sdk/releases/tag/wasi-sdk-25
set -e
cd "$(dirname "$0")"
WASI_SDK="${WASI_SDK:-$HOME/opt/wasi-sdk-25.0-x86_64-linux}"
if [ ! -x "$WASI_SDK/bin/clang" ]; then
    echo "build-wasm.sh: set WASI_SDK to an unpacked wasi-sdk (wasi-sdk-25.0-x86_64-linux from the release above)" >&2
    exit 1
fi
CC="$WASI_SDK/bin/clang --target=wasm32-wasip1 --sysroot=$WASI_SDK/share/wasi-sysroot"
CFLAGS="-O3 -msimd128 -flto -funroll-loops -ffast-math -D_WASI_EMULATED_GETPID"
LIBS="-lm -lwasi-emulated-getpid"

$CC $CFLAGS -mexec-model=reactor -DPPAT_NO_FILE_IO -DTOROGO_WASM_FATAL -Wl,--strip-all \
    -o ../ppat-playout.wasm game2.c ppat.c wasm_ppat.c $LIBS
$CC $CFLAGS -mexec-model=reactor -DPPAT_NO_FILE_IO -DTOROGO_WASM_FATAL -DWASM_PPAT_TEST -Wl,--strip-all \
    -o ppat-playout-test.wasm game2.c ppat.c wasm_ppat.c $LIBS
$CC $CFLAGS -Wl,-z,stack-size=8388608 -Wl,--strip-all \
    -o bench_ppat.wasm game2.c ppat.c bench_ppat.c $LIBS

# The value net: vnet_encode's ladder reads keep a Ladder2Result per point on
# the stack, so 1 MB of stack.
$CC $CFLAGS -mexec-model=reactor -DVNET_NO_FILE_IO -DTOROGO_WASM_FATAL -Wl,-z,stack-size=1048576 -Wl,--strip-all \
    -o ../vnet-eval.wasm game2.c game3.c ladder2.c cnn.c wasm_vnet.c $LIBS

echo "built ../ppat-playout.wasm, ppat-playout-test.wasm, bench_ppat.wasm, ../vnet-eval.wasm"
