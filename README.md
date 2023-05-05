# OpenMP_Parallel_Programming
The "mat.c" code multiplies two matrices using cache blocking and OpenMP parallelization, then it measures the execution time and prints the result matrix.  
In "piEstimation.c", OpenMP parallelization is implemented to estimate the value of pi using a Monte Carlo simulation, where random points are generated within a unit square and the ratio of the number of points inside a quarter-circle to the total number of points is used to approximate pi. The code uses parallelization to split the generation of random points across multiple threads to speed up the simulation. The reduction clause is used to combine the results from each thread to get the final pi estimate. The code also allows for command-line arguments to specify the number of points and the number of threads to use.  
NB :  To run this code on a linux distrubution :   
gcc  mat.c -fopenmp -o mat -lgomp  
Then  
./mat
