#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "utils/vec3.h"
#include "render/ray.h"


class Camera
{
public:
	__device__ Camera(vec3 lookFrom, vec3 lookAt, vec3 vUp, const float focusDist, const float aperture,
		const float vFov = 60.0f, const float aspectRatio = 16.0f / 9.0f);

	__device__ Ray GetRay(float u, float v, curandState* localRandState) const;

private:
	vec3 m_Origin;
	vec3 m_Horizontal;
	vec3 m_Vertical;
	vec3 m_LowerLeft; // lower left corner

	float m_LensRadius;
	vec3 m_W, m_U, m_V;
};
