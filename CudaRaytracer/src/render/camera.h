#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "utils/vec3.h"
#include "render/ray.h"


class Camera
{
public:
	__device__ Camera(float aspectRatio = 16.0f/9.0f)
		:m_Origin(0.0f, 0.0f, 0.0f)
	{
		float focalLength    = 1.0f;
		float viewportHeight = 2.0f;
		float viewportWidth  = aspectRatio * viewportHeight;

		m_Horizontal      = vec3(viewportWidth, 0.0f, 0.0f);
		m_Vertical        = vec3(0.0f, viewportHeight, 0.0f);
		m_LowerLeftCorner = m_Origin - m_Horizontal * 0.5f - m_Vertical * 0.5f - vec3(0.0f, 0.0f, focalLength);
	}

	__device__ inline Ray GetRay(float u, float v) const
	{
		return Ray(m_Origin, m_LowerLeftCorner + u * m_Horizontal + v * m_Vertical - m_Origin);
	}

private:
	vec3 m_Origin;
	vec3 m_Horizontal;
	vec3 m_Vertical;
	vec3 m_LowerLeftCorner;
};
