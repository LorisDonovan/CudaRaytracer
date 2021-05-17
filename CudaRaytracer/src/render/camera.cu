#include "camera.h"

#include "utils/utils.h"

__device__ vec3 RandomInUnitDisk(curandState* locaRandState)
{
	vec3 p;
	do
	{
		p = 2.0f * vec3(curand_uniform(locaRandState), curand_uniform(locaRandState), 0) - vec3(1, 1, 0);
	} while (Dot(p, p) >= 1.0f);
	return p;
}


__device__ Camera::Camera(vec3 lookFrom, vec3 lookAt, vec3 vUp, const float focusDist, const float aperture,
	const float vFov,      const float aspectRatio)
{
	float theta = DegreesToRadian(vFov);
	float h     = std::tan(theta / 2.0f);
	float focalLength    = 1.0f;
	float viewportHeight = 2.0f * h;
	float viewportWidth  = aspectRatio * viewportHeight;

	m_W = UnitVector(lookFrom - lookAt);
	m_U = UnitVector(Cross(vUp, m_W));
	m_V = Cross(m_W, m_U);

	m_Origin = lookFrom;
	m_Horizontal = focusDist * viewportWidth  * m_U;
	m_Vertical   = focusDist * viewportHeight * m_V;
	m_LowerLeft  = m_Origin - m_Horizontal * 0.5f - m_Vertical * 0.5f - focusDist * m_W;

	m_LensRadius = aperture / 2.0f; // if m_LensRadius = 0, no blur
}

__device__ Ray Camera::GetRay(float s, float t, curandState* localRandState) const
{
	// for defocus blur, generate random scene rays from a disk at lookFrom to the focus plane
	// blur depends on the radius of the disk and the focus distance
	vec3 rd = m_LensRadius * RandomInUnitDisk(localRandState);
	vec3 offset = m_U * rd.x() + m_V * rd.y();
	return Ray(m_Origin + offset, m_LowerLeft + s * m_Horizontal + t * m_Vertical - m_Origin - offset);
}
