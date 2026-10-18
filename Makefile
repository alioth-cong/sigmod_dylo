CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17
DIR := harness/cpp/dylo
ABLATIONS := selector selection_timing deferral_trace seed_dispersion marginal_bucket greedy
.PHONY: all ablations check-artifact case-study-audit clean
all: $(DIR)/dylo_driver_std4 $(DIR)/selector_verify
$(DIR)/dylo_driver_std4: $(DIR)/driver_std4.cxx $(wildcard $(DIR)/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $<
$(DIR)/selector_verify: $(DIR)/selector_verify.cxx $(wildcard $(DIR)/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $<
$(DIR)/driver_%.cxx: $(DIR)/ablations/%.patch $(DIR)/driver_std4.cxx
	patch --silent --output=$@ $(DIR)/driver_std4.cxx $<
$(DIR)/dylo_driver_%: $(DIR)/driver_%.cxx $(wildcard $(DIR)/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $<
ablations: $(addprefix $(DIR)/dylo_driver_,$(ABLATIONS))
check-artifact:
	python3 scripts/validate_artifact.py
case-study-audit:
	python3 experiments/case_study/audit.py
clean:
	rm -f $(DIR)/dylo_driver_* $(DIR)/selector_verify $(addprefix $(DIR)/driver_,$(addsuffix .cxx,$(ABLATIONS)))
