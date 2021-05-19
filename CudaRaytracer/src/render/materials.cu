#include "materials.h"

#include "utils/utils.h"


// ----------Utility functions-----------------------------
__device__ inline vec3 Reflect(const vec3& rayDir, const vec3& normal)
{
	return rayDir - 2.0f * Dot(rayDir, normal) * normal;
}

__device__ inline float Schlick(float cosine, float refractiveIdx)
{
	float r0 = (1.0f - refractiveIdx) / (1.0f + refractiveIdx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5.0f);
}

__device__ inline bool Refract(const vec3& unitVec, const vec3& normal, float etaIOverEtaT, vec3& refracted)
{
	float dt = Dot(unitVec, normal);
	float discriminant = 1.0f - etaIOverEtaT * etaIOverEtaT * (1.0f - dt * dt);
	if (discriminant > 0.0f)
	{
		refracted = etaIOverEtaT * (unitVec - normal * dt) - normal * std::sqrt(discriminant);
		return true;
	}
	
	return false;
}

// ----------Diffuse Material------------------------------
__device__ bool Lambertian::Scatter(const Ray& rayIn, const HitRecords& rec, vec3& attenuation,
	Ray& scatteredRay, curandState* localRandState) const
{
	vec3 target  = rec.Normal + RandomInUnitSphere(localRandState); // Scatter the rays in random direction
	scatteredRay = Ray(rec.Point, target);
	attenuation  = m_Albedo;
	return true;
}


// ----------Metal Material--------------------------------
__device__ bool Metal::Scatter(const Ray& rayIn, const HitRecords& rec, vec3& attenuation,
	Ray& scatteredRay, curandState* localRandState) const
{
	vec3 reflected = Reflect(rayIn.GetDirection(), rec.Normal);
	scatteredRay   = Ray(rec.Point, reflected + m_Fuzz * RandomInUnitSphere(localRandState));
	attenuation    = m_Albedo;
	return (Dot(scatteredRay.GetDirection(), rec.Normal) > 0.0f);
}


// ----------Glass Material--------------------------------
__device__ bool Dielectric::Scatter(const Ray& rayIn, const HitRecords& rec, vec3& attenuation,
	Ray& scatteredRay, curandState* localRandState) const
{
	vec3 refracted;
	vec3 reflected = Reflect(rayIn.GetDirection(), rec.Normal);
	attenuation    = vec3(1.0f, 1.0f, 1.0f);

	float etaIOverEtaT = rec.IsFrontFace ? (1.0f / m_RefractiveIndex) : m_RefractiveIndex; // eta incident over eta transmitted
	float dt = Dot(rayIn.GetDirection(), rec.Normal);
	float cosine = rec.IsFrontFace ? -dt : std::sqrt(1.0f - m_RefractiveIndex * m_RefractiveIndex * 1.0f - dt * dt);
	float reflectProb;

	if (Refract(rayIn.GetDirection(), rec.Normal, etaIOverEtaT, refracted))
		reflectProb = Schlick(cosine, m_RefractiveIndex);
	else
		reflectProb = 1.0f;

	if (curand_uniform(localRandState) < reflectProb)
		scatteredRay = Ray(rec.Point, reflected);
	else
		scatteredRay = Ray(rec.Point, refracted);

	return true;
}
