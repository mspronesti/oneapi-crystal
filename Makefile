all:
	mkdir -p build && cd build && cmake .. && make

.PHONY: all_queries 
all_queries:
	mkdir -p build && cd build && cmake .. -DBUILD_QUERIES=ON && make 