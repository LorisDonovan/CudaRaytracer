#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ray.h"
#include "utils/vec3.h"
#include "hittable.h"


class Materials
{
public:
	__device__ virtual bool Scatter(const Ray& rayIn, const HitRecords& rec, vec3& attenuation, 
		Ray& scatteredRay, curandState* localRandState) const = 0;
};


// ----------Diffuse Material------------------------------
class Lambertian : public Materials
{
public:
	__device__ Lambertian(const vec3& albedo)
		: m_Albedo(albedo) {}

	__device__ bool Scatter(const Ray& rayIn, const HitRecords& rec, vec3& attenuation, 
		Ray& scatteredRay, curandState* localRandState) const override;

private:
	vec3 m_Albedo;
};


// ----------Metal Material--------------------------------
class Metal : public Materials
{
public:
	__device__ Metal(const vec3& albedo, float fuzz = 0.0f)
		: m_Albedo(albedo), m_Fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

	__device__ bool Scatter(const Ray& rayIn, const HitRecords& rec, vec3& attenuation,
		Ray& scatteredRay, curandState* localRandState) const override;

private:
	vec3 m_Albedo;
	float m_Fuzz;
};


// ----------Glass Material--------------------------------
class Dielectric : public Materials
{
public:
	__device__ Dielectric(float refractiveIndex)
		: m_RefractiveIndex(refractiveIndex) {}

	__device__ bool Scatter(const Ray& rayIn, const HitRecords& rec, vec3& attenuation,
		Ray& scatteredRay, curandState* localRandState) const override;

private:
	float m_RefractiveIndex;

};
