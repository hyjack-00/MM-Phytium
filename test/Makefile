
ARMPL_PREFFIX = /opt/arm/armpl_23.10_gcc-9.3

all:
	@echo "doing nothing"

eigen:
	g++ -o eigen_test.exe kernels_eigen_test.cpp -I/usr/local/include/eigen3 -std=c++11 -O3 -march=native

armpl:
	g++ kernels_armpl_test.cpp -o armpl_test.exe -I$(ARMPL_PREFFIX)/include -L$(ARMPL_PREFFIX)/lib -larmpl -std=c++11 -O3 -march=native -lamath -lastring

libshalom:
	g++ -O3 -I../libShalom-full/