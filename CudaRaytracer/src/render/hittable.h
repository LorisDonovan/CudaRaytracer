#pragma once

#include "ray.h"

class Materials;

struct HitRecords
{
	vec3 Point;
	vec3 Normal;
	float t = 0.0f;
	bool IsFrontFace    = false;
	Materials* Material = nullptr;

	__device__ inline void SetFrontNormal(const Ray& ray, const vec3& outwardNormal)
	{
		IsFrontFace = Dot(ray.GetDirection(), outwardNormal) < 0.0f;
		Normal = IsFrontFace ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{
public:
	__device__ virtual bool Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const = 0;
};
