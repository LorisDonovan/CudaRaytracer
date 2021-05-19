#pragma once

#include "utils/vec3.h"

class Ray
{
public:
	__device__ Ray() {}
	__device__ Ray(const vec3& origin, const vec3& direction)
		:m_Origin(origin), m_Direction(UnitVector(direction)) {}

	__device__ inline vec3 GetOrigin()      const { return m_Origin; }
	__device__ inline vec3 GetDirection()   const { return m_Direction; }

	__device__ inline vec3 PointAt(float t) const { return m_Origin + t * m_Direction; } // P(t) = A + tb

private:
	vec3 m_Origin;
	vec3 m_Direction;
};
