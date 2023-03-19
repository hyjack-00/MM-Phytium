## Make 太复杂，暂时还是用 CMake

# CXX = g++
# CXXLIB = -fopenmp
# CXXFLAGS = -std=c++11 -O2 -falign-loops -march=native -ffast-math $(CXXLIB)

# CURRENT = bin/kernels_test
# TARGETS = bin/kernels_test

# ALL: build
# b build: $(TARGETS)
# 	@mkdir -p build bin
# r run: build
# 	./bin/$(CURRENT)


# bin/kernels_test: build/kernels_test.o build/mk_s32.o
# 	$(CXX) $(CXXFLAGS) $< -o $@


# SRCS += $(wildcard ./microkernels/*.cpp)
# SRCS += $(wildcard ./test/*.cpp)
# OBJS =  $(patsubst %.cpp, ./build/%.o, $(SRCS))


# %.o: %.cpp
# 	$(CXX) -c $(CXXFLAGS) $< -o $@

# # build/kernels_test.o: test/kernels_test.cpp
# # 	$(CXX) -c $(CXXFLAGS) $< -o $@
# # build/mk_s32.o: microkernels/mk_s32.cpp
# # 	$(CXX) -c $(CXXFLAGS) $< -o $@

# clean:
# 	rm -f build/*
# 	rm -f bin/*