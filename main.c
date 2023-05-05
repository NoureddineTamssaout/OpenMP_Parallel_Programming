#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define N 60
#define M 60
#define P 60
#define BLOCK_SIZE 32

int main() {
    double start, end;
    double execution_time;
    start = omp_get_wtime();

    double mat1[N][M], mat2[M][P], mul[N][P];
    int i, j, k, ii, jj, kk;

    // Initialize matrices in parallel
    #pragma omp parallel for private(j)
    for(i = 0; i < N; i++) {
        for(j = 0; j < M; j++) {
            mat1[i][j] = i*10.0 + j;
        }
    }

    #pragma omp parallel for private(j)
    for(i = 0; i < M; i++) {
        for(j = 0; j < P; j++) {
            mat2[i][j] = i*10.0 + j;
        }
    }

    // Matrix multiplication with cache blocking
    #pragma omp parallel for private(jj, kk, ii)
    for(i = 0; i < N; i += BLOCK_SIZE) {
        for(j = 0; j < P; j += BLOCK_SIZE) {
            for(k = 0; k < M; k += BLOCK_SIZE) {
                for(ii = i; ii < i + BLOCK_SIZE; ii++) {
                    for(jj = j; jj < j + BLOCK_SIZE; jj++) {
                        double sum = 0.0;
                        for(kk = k; kk < k + BLOCK_SIZE; kk++) {
                            sum += mat1[ii][kk] * mat2[kk][jj];
                        }
                        mul[ii][jj] += sum;
                    }
                }
            }
        }
    }

    // Print result matrix
    for(i = 0; i < N; i++) {
        for(j = 0; j < P; j++) {
            printf("%f ", mul[i][j]);
        }
        printf("\n");
    }

    end = omp_get_wtime();
    double duration = (double)(end - start);

    printf("execution time is : %f secondes \n", duration);

    return 0;
}
