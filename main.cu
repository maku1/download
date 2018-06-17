/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <time.h>
#include "support.h"
#include "kernel.cu"

//block size
#define  BLOCK_SIZE 512
#define  VECTOR_SIZE 100

int main(int argc, char**argv) {
   printf("copy\n");
    Timer timer;
    cudaError_t cuda_ret;
    time_t t;


    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    
    fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1)
    {
        n = 10000;
    }
    else if(argc == 2)
    {
        n = atoi(argv[1]);
    }
    else
    {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }
    
    /* Intializes random number generator */
    srand((unsigned) time(&t));    
    

    float* A_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++)
    {
    //Allocates a vctor with random float entries
       A_h[i] = (rand()%100)/100.00;
    }

    float* B_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++)
    {
        B_h[i] = (rand()%100)/100.00;
    }

    float* C_h = (float*) malloc( sizeof(float)*n );

    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables...");
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    // cudaError_t cudaMalloc	(	void ** 	devPtr,size_t 	size)
    //Allocates size bytes of linear memory on the device and
    //returns in *devPtr a pointer to the allocated memory.
    //The allocated memory is suitably aligned for any kind of variable.
    //The memory is not cleared. cudaMalloc() returns
    //cudaErrorMemoryAllocation in case of failure.
    cudaMalloc*(&a, n*sizeof(float));
    cudaMalloc*(&b, n*sizeof(float));
    cudaMalloc*(&c, n*sizeof(float));









    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device...");
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
//cudaError_t cudaMemcpy(void * 	dst, const void * 	src, size_t	count,enum cudaMemcpyKind 	kind)

//Copies count bytes from the memory area pointed to by src to the memory area pointed to by dst,
//where kind is one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
//cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice,
//and specifies the direction of the copy. The memory areas may not overlap.
//Calling cudaMemcpy() with dst and src pointers that do not match
//the direction of the copy results in an undefined behavior.

     float* a , b, c;
     cudaMemcpy(a, A_h,  n*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(b, B_h,  n*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(c, C_h, n*sizeof(float), cudaMemcpyDeviceToHost);
     
     

    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel...");
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
  

   vecAddKernel<<< 1,n >>>(A_h, B_h, C_h, n);
   
   


    cuda_ret = cudaDeviceSynchronize();
	 if(cuda_ret != cudaSuccess)
  {
     FATAL("Unable to launch kernel");
  }
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
     cudaMemcpy(A_h, a,  n*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(B_h, b,  n*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(C_h, c,  n*sizeof(float), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);


    return 0;

}

