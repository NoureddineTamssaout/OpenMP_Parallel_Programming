#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define MAX_THREADS 8

double estimate_pi(long long n) {
    int i, count = 0;
    double x, y, pi;
    unsigned int seed = (unsigned) time(NULL);

    #pragma omp parallel for reduction(+:count) private(x, y) shared(seed)
    for (i = 0; i < n; i++) {
        x = (double) rand_r(&seed) / RAND_MAX;
        y = (double) rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            count++;
        }
    }

    pi = 4.0 * (double) count / (double) n;
    return pi;
}

int main(int argc, char *argv[]) {
    long long n = 1000000000;
    int i, num_threads = MAX_THREADS;
    double pi, sum = 0.0, start, end;

    if (argc > 1) {
        n = atoll(argv[1]);
    }

    if (argc > 2) {
        num_threads = atoi(argv[2]);
        if (num_threads > MAX_THREADS) {
            num_threads = MAX_THREADS;
        }
    }

    omp_set_num_threads(num_threads);

    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < num_threads; i++) {
        sum += estimate_pi(n / num_threads);
    }

    pi = sum / num_threads;

    end = omp_get_wtime();

    printf("pi = %.16f (calculated with %lld points)\n", pi, n);
    printf("Time: %f seconds\n", end - start);

    return 0;
}