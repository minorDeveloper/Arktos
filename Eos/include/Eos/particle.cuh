//
// Created by samlane321 on 30/03/2021.
//

#ifndef CUDA_PARTICLE_H
#define CUDA_PARTICLE_H

#include "v3.cuh"

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


#endif //CUDA_PARTICLE_H
