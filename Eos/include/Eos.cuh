//
// Created by samlane321 on 07/04/2021.
//

#ifndef ARKTOS_EOS_CUH
#define ARKTOS_EOS_CUH

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "Eos/particle.cuh"


__global__
void advanceParticles(float dt, particle * pArray, int nParticles)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nParticles) { pArray[idx].advance(dt); }
}

int run(int argc, char ** argv) {
    int n = 1000000;
    if (argc > 1) { n = atoi(argv[1]); }     // Number of particles
    if (argc > 2) { srand(atoi(argv[2])); } // Random seed

    particle *pArray = new particle[n];
    particle *devPArray = NULL;
    cudaMalloc(&devPArray, n * sizeof(particle));
    cudaMemcpy(devPArray, pArray, n * sizeof(particle), cudaMemcpyHostToDevice);
    for (int i = 0; i < 100; i++) {   // Random distance each step
        float dt = (float) rand() / (float) RAND_MAX;
        advanceParticles<<< 1 + n / 256, 256>>>(dt, devPArray, n);
        cudaDeviceSynchronize();
    }
    return 0;
}

#endif//ARKTOS_EOS_CUH
