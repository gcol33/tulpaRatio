# Makefile for ratiod C++ tests (Catch2)
# Run from src/tests directory

# On Windows with Rtools45
CXX ?= g++
ifeq ($(OS),Windows_NT)
  CXX = C:/rtools45/x86_64-w64-mingw32.static.posix/bin/g++.exe
endif

# The package headers under test are header-only but not self-contained: they
# reach for R, Rcpp, RcppEigen and tulpa. Ask the installed R where those live
# rather than hardcoding a library path, so the suite compiles against the same
# headers the package is built against.
RSCRIPT ?= Rscript
RBIN ?= R
R_INCLUDE := $(shell $(RSCRIPT) -e "cat(shQuote(R.home('include')))")
PKG_INCLUDE := $(shell $(RSCRIPT) -e "cat(paste0('-I', shQuote(sapply(c('Rcpp','RcppEigen','tulpa'), function(p) system.file('include', package = p)))), sep=' ')")
R_LDFLAGS := $(shell $(RBIN) CMD config --ldflags)

CXXFLAGS = -std=c++17 -Wall -Wextra -I.. -I. -I$(R_INCLUDE) $(PKG_INCLUDE)
LDLIBS = $(R_LDFLAGS)

# Coverage flags
COV_FLAGS = --coverage -fprofile-arcs -ftest-coverage -O0 -g

TEST_SOURCES = test_main.cpp test_linalg.cpp test_zi.cpp

# Output executable
TEST_EXE = run_tests
TEST_EXE_COV = run_tests_cov

.PHONY: all test coverage clean

# Build and run tests (no coverage)
all: $(TEST_EXE)
	./$(TEST_EXE)

$(TEST_EXE): $(TEST_SOURCES)
	$(CXX) $(CXXFLAGS) -O2 -o $@ $^ $(LDLIBS)

# Build and run tests with coverage
coverage: $(TEST_EXE_COV)
	./$(TEST_EXE_COV)
	@echo ""
	@echo "=== Coverage Summary ==="
	gcov -r test_linalg.cpp test_zi.cpp 2>/dev/null | grep -A1 "File.*\.h"
	@echo ""
	@echo "Run 'make lcov' to generate HTML report"

$(TEST_EXE_COV): $(TEST_SOURCES)
	$(CXX) $(CXXFLAGS) $(COV_FLAGS) -o $@ $^ $(LDLIBS)

# Generate HTML coverage report using lcov
lcov: coverage
	lcov --capture --directory . --output-file coverage.info --rc lcov_branch_coverage=1
	lcov --remove coverage.info '/usr/*' '*/catch.hpp' --output-file coverage.info
	genhtml coverage.info --output-directory coverage_html --branch-coverage
	@echo "Coverage report: coverage_html/index.html"

# Run specific test file
test-linalg: $(TEST_EXE)
	./$(TEST_EXE) "[linalg]"

test-zi: $(TEST_EXE)
	./$(TEST_EXE) "[zi]"


# Verbose output
verbose: $(TEST_EXE)
	./$(TEST_EXE) -s

# List all tests
list: $(TEST_EXE)
	./$(TEST_EXE) --list-tests

# Clean build artifacts
clean:
	rm -f $(TEST_EXE) $(TEST_EXE_COV)
	rm -f *.gcno *.gcda *.gcov
	rm -f coverage.info
	rm -rf coverage_html
