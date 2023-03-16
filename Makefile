CC	= g++
CFLAGS	= -std=c++11 -O3 -falign-loops -march=native -ffast-math

all: mygemm fpdiff

mygemm: gemm.o
	$(CC) -o $@ $^ $(CFLAGS)

gemm.o: gemm.cpp
	$(CC) -c $(CFLAGS) $<

fpdiff: fpdiff.cpp
	$(CC) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o
	rm -f mygemm
	rm -f fpdiff