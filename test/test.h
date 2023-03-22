#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

using std::string;
using std::cout;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;
#define Dur(start,end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
