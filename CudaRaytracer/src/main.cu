#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "opengl/windowInit.h"
#include "opengl/screen.h"

#include "render/ray.h"
#include "render/hittable.h"
#include "render/hittableList.h"
#include "render/sphere.h"

#include "utils/vec3.h"
#include "utils/timer.h"
#include "utils/utils.h"


// Window property
constexpr float aspectRatio = 16.0f / 9.0f;
const  uint32_t height = 540;
const  uint32_t width  = static_cast<uint32_t>(height * aspectRatio);


// ----------Raytracer-------------------------------------
__global__ void CreateWorld(Hittable** list, Hittable** world);
__global__ void FreeWorld(Hittable** list, Hittable** world);
__global__ void Render(cudaSurfaceObject_t surfaceObj, Hittable** hittable, vec3 origin, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical);
__device__ vec3 RayColor(const Ray& ray, Hittable** hittable);


int main(int argc, char** argv)
{
	// CUDA resources
	cudaResourceDesc       resourceDesc;
	cudaGraphicsResource_t textureResource;
	cudaArray_t            textureArray;
	cudaSurfaceObject_t    surfaceObj = 0;

	// Camera and viewport
	float focalLength    = 1.0f;
	float viewportHeight = 2.0f;
	float viewportWidth  = aspectRatio * viewportHeight;
	vec3  origin(0.0f, 0.0f, 0.0f);
	vec3  horizontal(viewportWidth, 0.0f, 0.0f);
	vec3  vertical(0.0f, viewportHeight,  0.0f);
	vec3  lowerLeftCorner = origin - horizontal * 0.5f - vertical * 0.5f - vec3(0.0f, 0.0f, focalLength);

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
	Hittable** d_List;
	cudaCheckErrors(cudaMalloc((void**)&d_List, 2 * sizeof(Hittable*)));
	Hittable** d_World;
	cudaCheckErrors(cudaMalloc((void**)&d_World, sizeof(Hittable*)));
	CreateWorld<<<1, 1>>>(d_List, d_World);
	cudaCheckErrors(cudaGetLastError());
	cudaCheckErrors(cudaDeviceSynchronize());

	// CUDA kernel thread layout
	int32_t numThreads = 32;
	dim3 blocks((width + numThreads - 1) / numThreads, (height + numThreads - 1) / numThreads);
	dim3 threads(numThreads, numThreads);

	{
		Timer t; // Starts a timer when created and stops when destroyed
		// CUDA register and create surface object resource
		cudaCheckErrors(cudaGraphicsMapResources(1, &textureResource));
		cudaCheckErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0));
		resourceDesc.res.array.array = textureArray;
		cudaCheckErrors(cudaCreateSurfaceObject(&surfaceObj, &resourceDesc));
		Render<<<blocks, threads>>>(surfaceObj, d_World, origin, lowerLeftCorner, horizontal, vertical);
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
	FreeWorld<<<1, 1>>>(d_List, d_World);
	cudaCheckErrors(cudaGetLastError());

	cudaCheckErrors(cudaFree(d_List));
	cudaCheckErrors(cudaFree(d_World));
	Cleanup(quadVA, quadVB, textureID, shaderID);
	glfwTerminate();
	return 0;
}


// ----------Raytracer-------------------------------------
__global__ void CreateWorld(Hittable** list, Hittable** world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		list[0] = new Sphere(vec3(0.0f,    0.0f, -1.0f),   0.5f);
		list[1] = new Sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
		*world  = new HittableList(list, 2);
	}
}

__global__ void FreeWorld(Hittable** list, Hittable** world)
{
	delete list[0];
	delete list[1];
	delete *world;
}

__global__ void Render(cudaSurfaceObject_t surfaceObj, Hittable** world, 
	vec3 origin, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if ((x >= width) || (y >= height))
		return;

	// Offset values to move the ray across the screen
	float u    = float(x) / float(width);
	float v    = float(y) / float(height);
	// Calculate color
	Ray ray(origin, lowerLeftCorner + u * horizontal + v * vertical - origin);
	vec3 color = RayColor(ray, world);
	uint8_t r  = uint8_t(color.r() * 255);
	uint8_t g  = uint8_t(color.g() * 255);
	uint8_t b  = uint8_t(color.b() * 255);

	uchar4 data = make_uchar4(r, g, b, 255);
	surf2Dwrite(data, surfaceObj, x * sizeof(uchar4), y);
}

__device__ vec3 RayColor(const Ray& ray, Hittable** world)
{
	HitRecords rec;
	if ((*world)->Hit(ray, 0.001f, inf, rec))
		return 0.5f * vec3(rec.Normal + vec3(1.0f, 1.0f, 1.0f)); // Mapping to [0, 1]

	vec3 dir = ray.GetDirection();        // Direction of ray is a unit vector
	float t  = 0.5f * (dir.y() + 1.0f);   // Mapping y in the range [0, 1]
	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f); // Blend the background from blue to white vertically
}

