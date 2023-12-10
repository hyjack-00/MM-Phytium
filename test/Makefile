
ARMPL_PREFFIX = /opt/arm/armpl_23.10_gcc-9.3
LibShalom_INC = /home/kylin/MYLIB/install/SMM/include
LibShalom_LIB = /home/kylin/MYLIB/install/SMM/lib/libsmm.a

all:
	@echo "doing nothing"

eigen:
	g++ -o eigen_test.exe kernels_eigen_test.cpp -I/usr/local/include/eigen3 -std=c++11 -O3 -march=native

armpl:
	g++ -o armpl_test.exe kernels_armpl_test.cpp -I$(ARMPL_PREFFIX)/include -L$(ARMPL_PREFFIX)/lib -larmpl -std=c++11 -O3 -march=native -lamath -lastring

libshalom:
	g++ -o shalom_test.o kernels_libshalom_test.cpp -c -O3 -std=c++11 -I$(LibShalom_INC)
	g++ shalom_test.o $(LibShalom_LIB) -std=c++11 -o libshalom_test.exe