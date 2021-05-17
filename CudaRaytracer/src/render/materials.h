#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ray.h"
#include "utils/vec3.h"
#include "hittable.h"

struct HitRecords;

class Materials
{
public:
	__device__ virtual bool Scattered(const Ray& rayIn, const HitRecords& rec, vec3& attenuation, 
		Ray& scatteredRay, curandState* localRandState) const = 0;
};


// ----------Diffuse Material------------------------------
class Lambertian : public Materials
{
public:
	__device__ Lambertian(vec3 albedo)
		: m_Albedo(albedo) {}

	__device__ inline vec3 GetAlbedo() const { return m_Albedo; }

	__device__ bool Scattered(const Ray& rayIn, const HitRecords& rec, vec3& attenuation, 
		Ray& scatteredRay, curandState* localRandState) const override;

private:
	vec3 m_Albedo;
};
