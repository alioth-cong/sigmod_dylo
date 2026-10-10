# Builds DyLo and its experiment variants (C++17, no external dependencies).
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17
DIR      := harness/cpp/dylo
VARIANTS := std4 std4sel std4w seltime eps seedrep b3fix r2greedy
BINS     := $(addprefix $(DIR)/dylo_driver_,$(VARIANTS)) $(DIR)/selector_verify

all: $(BINS)

$(DIR)/dylo_driver_%: $(DIR)/driver_%.cxx $(wildcard $(DIR)/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(DIR)/selector_verify: $(DIR)/selector_verify.cxx $(wildcard $(DIR)/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
