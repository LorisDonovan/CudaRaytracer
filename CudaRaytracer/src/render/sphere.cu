#include "sphere.h"

__device__ bool Sphere::Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const
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
	float root  = (-b - sqrtd) / a; // Solution for t in sphere equation // quadratic equation
	if (root < tMin || root > tMax)
	{
		float root = (-b + sqrtd) / a;
		if (root < tMin || root > tMax)
			return false;
	}
	rec.t        = root;
	rec.Point    = ray.PointAt(rec.t);
	rec.Normal   = (rec.Point - m_Center) / m_Radius;
	rec.Material = m_Material;

	return true;
}

