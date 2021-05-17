#pragma once

#include "hittable.h"
#include "materials.h"

class Sphere : public Hittable
{
public:
	__device__ Sphere() = default;
	__device__ Sphere(const vec3& center, float radius, Materials* mat)
		: m_Center(center), m_Radius(radius), m_Material(mat) {}
	__device__ ~Sphere() { delete m_Material; }

	__device__ bool Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const override;

	__device__ inline vec3  GetCenter() const { return m_Center; }
	__device__ inline float GetRadius() const { return m_Radius; }

private:
	vec3 m_Center;
	float m_Radius;
	Materials* m_Material;
};


