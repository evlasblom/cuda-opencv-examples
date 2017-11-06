/**
 * @brief      Basic addition.
 *
 *             Here we do basic addition of two integers in a kernel for the
 *             GPU. For doing our first computation, we need to allocate and
 *             afterwards free  memory on the GPU. The syntax to do this is very
 *             similar to what we use in C or C++. Then, to be able to do
 *             multiple additions in parallel, we need to define the number of
 *             threads and blocks.
 *
 *             From: http://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf
 */
#include <iostream>
#include <string>


#define N (2048*2048)
#define M 512


/**
 * @brief      Kernel for basic addition.
 *
 * @param      a     Input integer 1
 * @param      b     Input integer 2
 * @param      c     Output integer
 */
__global__ void add(int *a, int *b, int *c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N) {
		c[i] = a[i] + b[i];
	}
}

int main(void) {

	// Declare copies on host and device
	int *h_a, *h_b, *h_c; 
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);

	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Setup input values
	h_a = (int *)malloc(size);
	h_b = (int *)malloc(size);
	h_c = (int *)malloc(size);
	for (int i = 0; i < N; ++i)	{
		h_a[i] = i;
		h_b[i] = i;
	}

	// Copy inputs to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<<(N+M-1)/M, M>>>(d_a, d_b, d_c);

	// Copy result back to host
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	// Output
	std::cout << "Massively parallel addition." << std::endl;
	std::cout << "Displaying first 10 out of " << N << " results: " << std::endl;
	for (int i = 0; i < 10; ++i)	{
		std::cout << h_a[i] << + " + " << h_b[i] << " = " << h_c[i] << std::endl;
	}

	// Cleanup
	free(h_a); free(h_b); free(h_c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;

}