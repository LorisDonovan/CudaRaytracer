#include "camera.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "utils/utils.h"


__device__ Camera::Camera(vec3 lookFrom, vec3 lookAt, vec3 vUp, const float focusDist, const float aperture,
	const float vFov, const float aspectRatio)
{
	float theta = DegreesToRadian(vFov);
	float h     = std::tan(theta / 2.0f);
	float viewportHeight = 2.0f * h;
	float viewportWidth  = aspectRatio * viewportHeight;

	m_FrontVec = UnitVector(lookFrom - lookAt);      // Front vector
	m_RightVec = UnitVector(Cross(vUp, m_FrontVec)); // Right vector
	m_UpVec    = Cross(m_FrontVec, m_RightVec);      // Up vector

	m_Origin     = lookFrom;
	m_Horizontal = focusDist * viewportWidth  * m_RightVec;
	m_Vertical   = focusDist * viewportHeight * m_UpVec;
	m_LowerLeft  = m_Origin - m_Horizontal * 0.5f - m_Vertical * 0.5f - focusDist * m_FrontVec;

	m_LensRadius = aperture / 2.0f; // if m_LensRadius = 0, no blur
}

__device__ Ray Camera::GetRay(float s, float t, curandState* localRandState) const
{
	// for defocus blur, generate random scene rays from a disk at lookFrom to the focus plane
	// blur depends on the radius of the disk and the focus distance
	vec3 rd = m_LensRadius * RandomInUnitDisk(localRandState);
	vec3 offset = m_RightVec * rd.x() + m_UpVec * rd.y();
	return Ray(m_Origin + offset, m_LowerLeft + s * m_Horizontal + t * m_Vertical - m_Origin - offset);
}
