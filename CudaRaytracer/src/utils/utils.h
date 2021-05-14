#include <limits>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#define cudaCheckErrors(val) CheckCuda(val, #val, __FILE__, __LINE__)

__device__ const float inf = std::numeric_limits<float>::infinity();
__device__ const float pi  = 3.1415926535897932385f;


__device__ inline float DegreesToRadian(float degrees) 
{
	return degrees * pi / 180.0f;
}

void    CheckCuda(cudaError_t result, const char* func, const char* filepath, const uint32_t line);
int32_t InitCudaDevice();
void    InitCudaTexture(cudaGraphicsResource_t& textureResource, cudaResourceDesc& resourceDesc, uint32_t textureID);
