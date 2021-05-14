#pragma once

#include "hittable.h"

class Sphere : public Hittable
{
public:
	__device__ Sphere() = default;
	__device__ Sphere(const vec3& center, float radius)
		: m_Center(center), m_Radius(radius) {}

	__device__ bool Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const override
	{
		vec3 oc = ray.GetOrigin() - m_Center;
		// Simplified Ray-Sphere intersection
		float a = ray.GetDirection().LengthSquared();
		float b = Dot(ray.GetDirection(), oc);
		float c = oc.LengthSquared() - m_Radius * m_Radius;
		float dis = b * b - a * c; // discriminant

		if (dis < 0.0f) // The ray didnt interesect the sphere
			return false;

		// Only take the nearest root that lies in the acceptable range (tMin, tMax)
		float sqrtd = std::sqrt(dis);
		float root = (-b - sqrtd) / a; // Solution for t in sphere equation // quadratic equation
		if (root < tMin || root > tMax)
		{
			float root = (-b + sqrtd) / a;
			if (root < tMin || root > tMax)
				return false;
		}
		rec.t = root;
		rec.Point = ray.PointAt(rec.t);
		rec.Normal = (rec.Point - m_Center) / m_Radius;

		return true;
	}

	__device__ inline vec3  GetCenter() const { return m_Center; }
	__device__ inline float GetRadius() const { return m_Radius; }

private:
	vec3 m_Center;
	float m_Radius;
};

