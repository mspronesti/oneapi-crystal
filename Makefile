all:
	mkdir -p build && cd build && cmake .. && make

.PHONY: queries 
queries:
	mkdir -p build && cd build && cmake .. -DBUILD_QUERIES=ON && make -j `nproc`

.PHONY: operators
operators:
	mkdir -p build && cd build && cmake .. -DBUILD_OPERATORS=ON && make -j `nproc`

