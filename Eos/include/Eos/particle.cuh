//
// Created by samlane321 on 03/04/2021.
//

#ifndef CUDA_LIB_PARTICLE_CUH
#define CUDA_LIB_PARTICLE_CUH


#include "Eos/v3.cuh"

class particle {
public:
    particle();
    __host__ __device__ void advance(float dist);
    const v3& getTotalDistance() const;

private:
    v3 position;
    v3 velocity;
    v3 totalDistance;
};


#endif //CUDA_LIB_PARTICLE_CUH
