CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17
HEADERS := $(wildcard src/*.hpp)

.PHONY: all test clean aliases
all: bin/dylo_std_corrected bin/dylo_std_reference bin/tests bin/selector_tests bin/hierarchy_tests bin/dp_cache_tests bin/selector_scaling aliases

bin:
	mkdir -p bin

bin/dylo_std_corrected: src/driver_std_corrected.cxx $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/dylo_std_reference: src/driver_std_reference.cxx $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/tests: tests.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/selector_tests: selector_tests.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/hierarchy_tests: hierarchy_tests.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/dp_cache_tests: dp_cache_tests.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/selector_scaling: selector_scaling.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

aliases: bin/dylo_std_corrected
	for alias in dylo_driver dylo_driver_std4 dylo_driver_std4sel dylo_driver_seltime dylo_driver_eps dylo_driver_seedrep dylo_driver_b3fix dylo_driver_r2greedy dylo_driver_w dylo_driver_std4w; do ln -sfn dylo_std_corrected bin/$$alias; done

test: all
	./bin/tests
	./bin/selector_tests
	./bin/hierarchy_tests
	./bin/dp_cache_tests
	./bin/selector_scaling selector_scaling.csv
	python test_features.py
	python run_small.py
	python test_deferred.py
	python test_cost_scope.py
	python test_adapter.py
	python test_report_validation.py
	python test_score_cost.py
	python test_ack.py

clean:
	rm -rf bin
