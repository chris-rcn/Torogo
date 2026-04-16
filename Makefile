CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Test targets
TEST_C_BIN = test-game3-c
TEST_C_SRC = test-game3-c.c game3.c
TEST_C_OBJ = $(TEST_C_SRC:.c=.o)

# Benchmark targets
BENCH_C_BIN = bench-game3-c
BENCH_C_SRC = bench-game3-c.c game3.c
BENCH_C_OBJ = $(BENCH_C_SRC:.c=.o)

# Default target
all: $(TEST_C_BIN) $(BENCH_C_BIN)

# Build C test executable
$(TEST_C_BIN): $(TEST_C_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build C benchmark executable
$(BENCH_C_BIN): $(BENCH_C_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile C source files
%.o: %.c
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

# Run benchmarks with timing
bench-timing: $(BENCH_C_BIN)
	time ./$(BENCH_C_BIN)

# Clean build artifacts
clean:
	rm -f $(TEST_C_BIN) $(BENCH_C_BIN) $(TEST_C_OBJ) $(BENCH_C_OBJ)

# Rebuild everything
rebuild: clean all

.PHONY: all test test-verbose clean rebuild
