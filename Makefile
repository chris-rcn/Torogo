CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -Ic/
LDFLAGS = -lm

# C source directory
C_DIR = c/

# Test targets
TEST_C_BIN = test-game3-c
TEST_C_SRC = $(C_DIR)test-game3-c.c $(C_DIR)game3.c
TEST_C_OBJ = test-game3-c.o game3.o

# Benchmark targets
BENCH_C_BIN = bench-game3-c
BENCH_C_SRC = $(C_DIR)bench-game3-c.c $(C_DIR)game3.c
BENCH_C_OBJ = bench-game3-c.o game3.o

BENCH_LADDER_BIN = bench-ladder-c
BENCH_LADDER_SRC = $(C_DIR)bench-ladder-c.c $(C_DIR)game3.c
BENCH_LADDER_OBJ = bench-ladder-c.o game3.o

# Default target
all: $(TEST_C_BIN) $(BENCH_C_BIN) $(BENCH_LADDER_BIN)

# Build C test executable
$(TEST_C_BIN): $(TEST_C_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build C benchmark executable
$(BENCH_C_BIN): $(BENCH_C_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build C ladder benchmark executable
$(BENCH_LADDER_BIN): $(BENCH_LADDER_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile C source files
%.o: $(C_DIR)%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run C tests
test: $(TEST_C_BIN)
	./$(TEST_C_BIN)

# Run with verbose output
test-verbose: $(TEST_C_BIN)
	./$(TEST_C_BIN) -v

# Run C benchmarks
bench: $(BENCH_C_BIN)
	./$(BENCH_C_BIN)

# Run ladder benchmarks
bench-ladder: $(BENCH_LADDER_BIN)
	./$(BENCH_LADDER_BIN)

# Run benchmarks with timing
bench-timing: $(BENCH_C_BIN)
	time ./$(BENCH_C_BIN)

# Run ladder benchmarks with timing
bench-ladder-timing: $(BENCH_LADDER_BIN)
	time ./$(BENCH_LADDER_BIN)

# Clean build artifacts
clean:
	rm -f $(TEST_C_BIN) $(BENCH_C_BIN) $(BENCH_LADDER_BIN) $(TEST_C_OBJ) $(BENCH_C_OBJ) $(BENCH_LADDER_OBJ)

# Rebuild everything
rebuild: clean all

.PHONY: all test test-verbose clean rebuild
