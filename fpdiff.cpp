#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, const char**argv) {
    FILE*you = fopen("result.dat", "r");
    FILE*ans = fopen("ref.dat", "r");
    bool y_eof = false;
    bool a_eof = false;
    while (true) {
        double a, b;
        int i, j;
        y_eof = fscanf(you, "%lf", &a) == -1;
        a_eof = fscanf(ans, "%lf", &b) == -1;
        if (y_eof || a_eof) {
            fclose(you);
            fclose(ans);
            if (!y_eof || !a_eof) {
                printf("length not match\n");
                return 0;
            }
            return 0;
        }
        if (fabs((a - b) / b) > 1e-5) {
            printf("y[%d]=%e, a[%d]=%e\n", i, a, j, b);
        }
    }
}
