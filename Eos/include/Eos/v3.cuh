//
// Created by samlane321 on 30/03/2021.
//

#ifndef CUDA_V3_H
#define CUDA_V3_H


class v3 {
public:
    float x;
    float y;
    float z;

    v3();
    v3(float xln, float yln, float zln);
    void randomize();
    __host__ __device__ void normalize();
    __host__ __device__ void scramble();
};


#endif //CUDA_V3_H
