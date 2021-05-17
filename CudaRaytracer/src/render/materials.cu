#include "materials.h"

// ----------Utility functions-----------------------------
__device__ vec3 RandomInUnitSphere(curandState* localRandState)
{
	vec3 p;
	do
	{
		p = 2.0f * vec3(curand_uniform(localRandState), curand_uniform(localRandState), curand_uniform(localRandState))
			- vec3(1.0f, 1.0f, 1.0f); // in the range of [-1, 1]
	} while (p.LengthSquared() >= 1.0f);

	return p;
}


// ----------Diffuse Material------------------------------
__device__ bool Lambertian::Scattered(const Ray& rayIn, const HitRecords& rec, vec3& attenuation,
	Ray& scatteredRay, curandState* localRandState) const
{
	vec3 target  = rec.Normal + RandomInUnitSphere(localRandState);
	scatteredRay = Ray(rec.Point, target);
	attenuation  = m_Albedo;
	return true;
}
