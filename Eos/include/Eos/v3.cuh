//
// Created by samlane321 on 03/04/2021.
//

#ifndef CUDA_LIB_V3_CUH
#define CUDA_LIB_V3_CUH


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


#endif //CUDA_LIB_V3_CUH
