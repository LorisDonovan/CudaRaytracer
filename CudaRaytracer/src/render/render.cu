#include "render.h"

#include <cstdint>

#include <device_launch_parameters.h>

#include "sphere.h"
#include "hittableList.h"
#include "materials.h"
#include "utils/utils.h"


__global__ void RenderInit(curandState* randState, const int32_t width, const int32_t height)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= width || y >= height)
		return;

	int32_t pixelIdx = x + y * width;
	// Random numbers for each thread
	curand_init(1984, pixelIdx, 0, &randState[pixelIdx]);
}

__global__ void Render(cudaSurfaceObject_t surfaceObj, Hittable** world, Camera** cam, curandState* randState, 
	const int32_t width, const int32_t height, const int32_t numSamples)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if ((x >= width) || (y >= height))
		return;

	int32_t pixelIdx = x + y * width;
	curandState localRandState = randState[pixelIdx];
	vec3 color(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < numSamples; i++)
	{
		// Offset values to move the ray across the screen
		float u = float(x + curand_uniform(&localRandState)) / float(width);
		float v = float(y + curand_uniform(&localRandState)) / float(height);
		color += RayColor((*cam)->GetRay(u, v, &localRandState), world, &localRandState, 50);
	}
	
	// Calculate color
	color /= float(numSamples);
	// Gamma correction
	uint8_t r = uint8_t(std::sqrt(color.r()) * 255);
	uint8_t g = uint8_t(std::sqrt(color.g()) * 255);
	uint8_t b = uint8_t(std::sqrt(color.b()) * 255);

	uchar4 data = make_uchar4(r, g, b, 255);
	surf2Dwrite(data, surfaceObj, x * sizeof(uchar4), y);
}

__device__ vec3 RayColor(const Ray& ray, Hittable** world, curandState* localRandState, const int32_t depth)
{
	Ray curRay = ray;
	vec3 curAttenuation(1.0f, 1.0f, 1.0f);
	for (int i = 0; i < depth; i++)
	{
		HitRecords rec;
		if ((*world)->Hit(curRay, 0.001f, inf, rec))
		{
			Ray scatteredRay;
			vec3 attenuation;
			if (rec.Material->Scatter(curRay, rec, attenuation, scatteredRay, localRandState))
			{
				curAttenuation *= attenuation;
				curRay = scatteredRay;
			}
			else
				return vec3(0.0f, 0.0f, 0.0f);
		}
		else
		{
			float t  = 0.5f * (curRay.GetDirection().y() + 1.0f); // Mapping y in the range [0, 1]
			// Blend the background from blue to white vertically
			vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
			return curAttenuation * c;
		}
	}

	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void CreateWorld(Camera** cam, Hittable** list, Hittable** world, const float aspectRatio, const int32_t numObj)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		list[0] = new Sphere(vec3( 0.0f, -100.5f, -1.0f), 100.0f, new Lambertian(vec3(rgb(142, 149, 105))));
		list[1] = new Sphere(vec3( 0.0f,    0.0f, -1.0f),   0.5f, new Lambertian(vec3(rgb(214, 181, 182))));
		list[2] = new Sphere(vec3(-1.0f,    0.0f, -1.0f),   0.5f, new Metal(vec3(rgb(50, 72, 93)), 0.05f));
		list[3] = new Sphere(vec3( 1.0f,    0.0f, -1.0f),   0.5f, new Dielectric(1.5f));
		*world  = new HittableList(list, numObj);
		// camera settings
		vec3  lookFrom  = vec3( 3.5f,  3.5f,  3.5f);
		vec3  lookAt    = vec3( 0.0f,  0.0f, -1.0f);
		vec3  vUp       = vec3( 0.0f,  1.0f,  0.0f);
		float focusDist = (lookFrom - lookAt).Length();
		float aperture  = 0.44f;
		float vFov      = 20.0f;

		*cam = new Camera(lookFrom, lookAt, vUp, focusDist, aperture, vFov, aspectRatio);
	}
}

__global__ void FreeWorld(Camera** cam, Hittable** list, Hittable** world, const int32_t numObj)
{
	for (int i = 0; i < numObj; i++)
		delete list[i];

	delete *world;
	delete *cam;
}
