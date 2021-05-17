#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "opengl/windowInit.h"
#include "opengl/screen.h"

#include "render/ray.h"
#include "render/hittable.h"
#include "render/hittableList.h"
#include "render/sphere.h"
#include "render/camera.h"

#include "utils/vec3.h"
#include "utils/timer.h"
#include "utils/utils.h"


// ----------Settings--------------------------------------
constexpr int32_t numThreadsX = 32;
constexpr int32_t numThreadsY = 32;
constexpr int32_t numSamples  = 256;
// Window settings
constexpr float aspectRatio = 16.0f / 9.0f;
constexpr uint32_t height   = 540;
constexpr uint32_t width    = static_cast<uint32_t>(height * aspectRatio);


// ----------Raytracer-------------------------------------
__device__ vec3 RayColor(const Ray& ray, Hittable** hittable, curandState* localRandState, const int32_t depth = 50);
__global__ void RenderInit(curandState* randState);
__global__ void Render(cudaSurfaceObject_t surfaceObj, Hittable** world, Camera** cam, curandState* randState);
__global__ void CreateWorld(Camera** cam, Hittable** list, Hittable** world);
__global__ void FreeWorld(Camera** cam, Hittable** list, Hittable** world, const int32_t numObj);


int main(int argc, char** argv)
{
	// CUDA resources
	cudaResourceDesc       resourceDesc;
	cudaGraphicsResource_t textureResource;
	cudaArray_t            textureArray;
	cudaSurfaceObject_t    surfaceObj = 0;

	// Initialize opengl and cuda interop
	GLFWwindow* window = InitWindow(width, height);
	int32_t cudaDevID  = InitCudaDevice();

	// Initialize vertex array and vertex buffer
	uint32_t quadVA, quadVB, shaderID;
	InitFbQuad(quadVA, quadVB, shaderID);

	// Initialize texture
	uint32_t textureID = InitGLTexture(width, height);
	InitCudaTexture(textureResource, resourceDesc, textureID);
	
	// Create Scene objects
	const int32_t numObj = 2;
	Camera** d_Cam;
	cudaCheckErrors(cudaMalloc((void**)&d_Cam, sizeof(Camera*)));
	Hittable** d_List;
	cudaCheckErrors(cudaMalloc((void**)&d_List, numObj * sizeof(Hittable*)));
	Hittable** d_World;
	cudaCheckErrors(cudaMalloc((void**)&d_World, sizeof(Hittable*)));
	CreateWorld<<<1, 1>>>(d_Cam, d_List, d_World);
	cudaCheckErrors(cudaGetLastError());
	cudaCheckErrors(cudaDeviceSynchronize());

	// CUDA kernel thread layout
	dim3 blocks((width + numThreadsX - 1) / numThreadsX, (height + numThreadsY - 1) / numThreadsY);
	dim3 threads(numThreadsX, numThreadsY);

	std::cout << "Rendering info:\n"
			  << "    Image Resolution: " << width       << "x" << height     << "\n"
			  << "    Thread dimension: " << numThreadsX << "x" << numThreadsY << "\n"
			  << "    Render samples  : " << numSamples << std::endl;
	
	// Initialize random numbers for Rendering
	curandState* d_RandState;
	{
		std::cout << "RenderInit: ";
		Timer t;
		cudaCheckErrors(cudaMalloc((void**)&d_RandState, width * height * sizeof(curandState)));
		RenderInit<<<blocks, threads>>>(d_RandState);
		cudaCheckErrors(cudaGetLastError());
		cudaCheckErrors(cudaDeviceSynchronize());
	}
	// Call Render function
	{
		std::cout << "Renderer   :";
		Timer t; // Starts a timer when created and stops when destroyed
		// CUDA register and create surface object resource
		cudaCheckErrors(cudaGraphicsMapResources(1, &textureResource));
		cudaCheckErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0));
		resourceDesc.res.array.array = textureArray;
		cudaCheckErrors(cudaCreateSurfaceObject(&surfaceObj, &resourceDesc));
		Render<<<blocks, threads>>>(surfaceObj, d_World, d_Cam, d_RandState);
		cudaCheckErrors(cudaGraphicsUnmapResources(1, &textureResource)); // sync cuda operations before graphics calls
		cudaCheckErrors(cudaGetLastError());
		cudaCheckErrors(cudaDeviceSynchronize());
	}

	while (!glfwWindowShouldClose(window))
	{
		// Render
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		ShaderBind(shaderID);
		glBindVertexArray(quadVA);
		TextureBind(textureID);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Cleanup
	FreeWorld<<<1, 1>>>(d_Cam, d_List, d_World, numObj);
	cudaCheckErrors(cudaGetLastError());

	cudaCheckErrors(cudaFree(d_List));
	cudaCheckErrors(cudaFree(d_World));
	cudaCheckErrors(cudaFree(d_Cam));
	Cleanup(quadVA, quadVB, textureID, shaderID);
	glfwTerminate();
	return 0;
}


// ----------Raytracer-------------------------------------
__global__ void RenderInit(curandState* randState)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= width || y >= height)
		return;

	int32_t pixelIdx = x + y * width;
	// Random numbers for each thread
	curand_init(1984, pixelIdx, 0, &randState[pixelIdx]);
}

__global__ void Render(cudaSurfaceObject_t surfaceObj, Hittable** world, Camera** cam, curandState* randState)
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
		color += RayColor((*cam)->GetRay(u, v), world, &localRandState, 50);
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
	float attenuation = 1.0f;
	for (int i = 0; i < depth; i++)
	{
		HitRecords rec;
		if ((*world)->Hit(curRay, 0.001f, inf, rec))
		{
			vec3 target = rec.Point + rec.Normal + RandomInUnitSphere(localRandState);
			curRay = Ray(rec.Point, target - rec.Point);
			attenuation *= 0.5f;
		}
		else
		{
			vec3 dir = ray.GetDirection();        // Direction of ray is a unit vector
			float t = 0.5f * (dir.y() + 1.0f);   // Mapping y in the range [0, 1]
			vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f); // Blend the background from blue to white vertically
			return attenuation * c;
		}
	}

	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void CreateWorld(Camera** cam, Hittable** list, Hittable** world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		list[0] = new Sphere(vec3(0.0f,    0.0f, -1.0f),   0.5f);
		list[1] = new Sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
		*world  = new HittableList(list, 2);
		*cam    = new Camera(aspectRatio);
	}
}

__global__ void FreeWorld(Camera** cam, Hittable** list, Hittable** world, const int32_t numObj)
{
	for (int i = 0; i < numObj; i++)
		delete list[i];

	delete *world;
	delete *cam;
}


