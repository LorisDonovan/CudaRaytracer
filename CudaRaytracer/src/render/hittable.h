#pragma once

#include "ray.h"

class Materials;

struct HitRecords
{
	vec3 Point;
	vec3 Normal;
	float t = 0.0f;
	Materials* Material;
};

class Hittable
{
public:
	__device__ virtual bool Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const = 0;
};
