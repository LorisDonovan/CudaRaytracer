#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "utils/vec3.h"
#include "render/hittable.h"
#include "render/camera.h"


__device__ vec3 RayColor(const Ray& ray, Hittable** hittable, curandState* localRandState, const int32_t depth = 50);

__global__ void RenderInit(curandState* randState, const int32_t width, const int32_t height);

__global__ void Render(cudaSurfaceObject_t surfaceObj, Hittable** world, Camera** cam, curandState* randState,
	const int32_t width, const int32_t height, const int32_t numSamples);

__global__ void CreateWorld(Camera** cam, Hittable** list, Hittable** world, const float aspectRatio, const int32_t numObj);

__global__ void FreeWorld(Camera** cam, Hittable** list, Hittable** world, const int32_t numObj);
