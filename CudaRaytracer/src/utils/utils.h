#pragma once 

#include <limits>
#include <iostream>

#include <cuda_runtime.h>

#include "utils/vec3.h"

#define cudaCheckErrors(val) CheckCuda(val, #val, __FILE__, __LINE__)


__device__ static const float inf = std::numeric_limits<float>::infinity();
__device__ static const float pi  = 3.1415926535897932385f;


inline void CheckCuda(cudaError_t result, const char* func, const char* file, const int32_t line)
{
	if (result)
	{
		printf("CUDA::ERROR_CODE::%d in function:'%s' file: '%s' line: %d\nERROR_DESCRIPTION:: %s\n",
			result, func, file, line, cudaGetErrorString(result));
		char ch = getchar();
		exit(-1);
	}
}

__device__ inline float DegreesToRadian(float degrees)
{
	return degrees * pi / 180.0f;
}

__device__ inline vec3 RandomInUnitDisk(curandState* locaRandState)
{
	vec3 p;
	do
	{
		p = 2.0f * vec3(curand_uniform(locaRandState), curand_uniform(locaRandState), 0.0f) - vec3(1.0f, 1.0f, 0.0f);
	} while (Dot(p, p) >= 1.0f);
	return p;
}

__device__ inline vec3 RandomInUnitSphere(curandState* localRandState)
{
	vec3 p;
	do
	{
		p = 2.0f * vec3(curand_uniform(localRandState), curand_uniform(localRandState), curand_uniform(localRandState))
			- vec3(1.0f, 1.0f, 1.0f); // in the range of [-1, 1]
	} while (p.LengthSquared() >= 1.0f);

	return p;
}


