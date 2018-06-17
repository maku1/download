/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
//__global__ prefixes a kernel the following function will run in parallel on the GPU(device)
//runs on the device
//is called from host code


__global__ void vecAddKernel(float* A, float* B, float* C, int n) {

    // Calculate global thread index based on the block and thread indices ----
    int i =  threadIdx.x;
    //INSERT KERNEL CODE HERE
    
    C[i] = A[i] + B[i];



    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE




}

