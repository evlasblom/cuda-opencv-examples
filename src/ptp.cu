/**
 * @brief      Usage of pointers on host and device.
 *
 *             In this example we allocate memory on the device for two pointers
 *             to floats, d_A and d_B, that point to the first element of a
 *             fixed block of n floats. This is an important first step to
 *             understanding how images are handles.
 *
 *             Then, we allocate memory on the device for a set of pointers to
 *             pointers to floats, d_X. We let those pointers to floats be equal
 *             to the first elements of the previously uploaded arrays. This
 *             allows us to access multiple arrays of floats from a single array
 *             of pointers. This example will be required later on when we
 *             handle vectors of images.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


/**
 * @brief      Kernel demonstrating usage of pointers.
 *
 * @param      d_X   Pointer to pointer to floats
 * @param      d_A   Pointer to floats
 * @param      d_B   Pointer to floats
 * @param[in]  n     Number of elements in input arrays
 */
__global__ void p_kernel(float ** d_X, float * d_A, float * d_B, int n){
  printf("Addresses:\n");
  printf("dX    = %p\n", d_X);
  printf("dA    = %p\n", d_A);
  printf("dB    = %p\n", d_B);
  printf("dX[0] = %p\n", d_X[0]);
  printf("dX[0] = %p\n", d_X[1]);

  float * devA  = d_X[0];
  float * devB  = d_X[1];

  printf("\nValues:\n");
  for (int i=0; i<n; i++)
  printf("A[%d] = %f\n", i, devA[i]);
  for (int i=0; i<n; i++)
    printf("B[%d] = %f\n", i, devB[i]);

}

int main(void) {
  // Declarations 
  const int n = 10;
  const int nn = n * sizeof(float);
  float * h_A;
  float * h_B;
  float * d_A;
  float * d_B;
  float ** hst_ptr;
  
  // Allocate space for h_A and h_B
  h_A = (float*)malloc(nn);
  h_B = (float*)malloc(nn);
	
  // Allocate space on the host for hst_ptr
  // as a mapped variable (so that the device can 
  // access it directly)  
  (cudaHostAlloc((void**)&hst_ptr, 2*sizeof(float*), cudaHostAllocMapped));

  for (int i=0; i<n; ++i) {
    h_A[i] = i + 1.0f;
    h_B[i] = 20.0f + i;
  }

  // Allocate space on the device for d_A and d_A
  (cudaMalloc((void**)&d_A, nn));
  (cudaMalloc((void**)&d_B, nn));

  (cudaMemcpy(d_A, h_A, nn, cudaMemcpyHostToDevice));
  (cudaMemcpy(d_B, h_B, nn, cudaMemcpyHostToDevice));

  hst_ptr[0]=d_A;
  hst_ptr[1]=d_B;

  p_kernel<<<1,1>>>(hst_ptr, d_A, d_B, n);

  // Free the resources.
  if (hst_ptr) (cudaFreeHost(hst_ptr));
  if (d_A) (cudaFree(d_A));
  if (d_A) (cudaFree(d_B));
  if (h_A) free(h_A);
  if (h_B) free(h_B);

  return EXIT_SUCCESS;
}
