#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "core/cudaInit.h"
#include "core/windowInit.h"
#include "core/screen.h"

#include "render/hittable.h"
#include "render/camera.h"
#include "render/render.h"

#include "utils/timer.h"


// ----------Settings--------------------------------------
constexpr int32_t numThreadsX = 16;
constexpr int32_t numThreadsY = 16;
constexpr int32_t numSamples  = 32;
// Window settings
constexpr float aspectRatio = 16.0f / 9.0f;
constexpr uint32_t height   = 270;
constexpr uint32_t width    = static_cast<uint32_t>(height * aspectRatio);


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
	const int32_t numObj = 4;
	Camera** d_Cam;
	cudaCheckErrors(cudaMalloc((void**)&d_Cam, sizeof(Camera*)));
	Hittable** d_List;
	cudaCheckErrors(cudaMalloc((void**)&d_List, numObj * sizeof(Hittable*)));
	Hittable** d_World;
	cudaCheckErrors(cudaMalloc((void**)&d_World, sizeof(Hittable*)));
	CreateWorld<<<1, 1>>>(d_Cam, d_List, d_World, aspectRatio, numObj);
	cudaCheckErrors(cudaGetLastError());
	//cudaCheckErrors(cudaDeviceSynchronize());

	// CUDA kernel thread layout
	dim3 blocks((width + numThreadsX - 1) / numThreadsX, (height + numThreadsY - 1) / numThreadsY);
	dim3 threads(numThreadsX, numThreadsY);

	std::cout << "Rendering info:\n"
			  << "    Image Resolution: " << width      << "x" << height    << "\n"
			  << "    Thread dimension: " << threads.x  << "x" << threads.y << "\n"
			  << "    Block dimension : " << blocks.x   << "x" << blocks.y  << "\n"
			  << "    Render samples  : " << numSamples << std::endl;
	
	// Initialize random numbers for Rendering
	curandState* d_RandState;
	{	
		std::cout << "RenderInit: ";
		Timer t;
		cudaCheckErrors(cudaMalloc((void**)&d_RandState, width * height * sizeof(curandState)));
		RenderInit<<<blocks, threads>>>(d_RandState, width, height);
		cudaCheckErrors(cudaGetLastError());
		//cudaCheckErrors(cudaDeviceSynchronize());
	}
	// Call Render function
	{
		std::cout << "Renderer  : ";
		Timer t; // Starts a timer when created and stops when destroyed
		// CUDA register and create surface object resource
		cudaCheckErrors(cudaGraphicsMapResources(1, &textureResource));
		cudaCheckErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0));
		resourceDesc.res.array.array = textureArray;
		cudaCheckErrors(cudaCreateSurfaceObject(&surfaceObj, &resourceDesc));
		Render<<<blocks, threads>>>(surfaceObj, d_World, d_Cam, d_RandState, width, height, numSamples);
		cudaCheckErrors(cudaGraphicsUnmapResources(1, &textureResource)); // sync cuda operations before graphics calls
		cudaCheckErrors(cudaGetLastError());
		//cudaCheckErrors(cudaDeviceSynchronize());
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
	cudaCheckErrors(cudaFree(d_RandState));
	Cleanup(quadVA, quadVB, textureID, shaderID);
	glfwTerminate();
	return 0;
}
