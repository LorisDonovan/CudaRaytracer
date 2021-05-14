#include "utils.h"

void CheckCuda(cudaError_t result, const char* func, const char* filepath, const uint32_t line)
{
	if (result)
	{
		std::cerr << "CUDA::ERROR:" << static_cast<uint32_t>(result) << " in file: \"" << filepath
			<< "\": line " << line << " - '" << func << "'" << std::endl;
		cudaDeviceReset();
		__debugbreak();
	}
}

int32_t InitCudaDevice()
{
	int32_t dev;
	cudaDeviceProp prop;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	// Compute capability >= 3.0
	prop.major = 3;
	prop.minor = 0;
	cudaCheckErrors(cudaChooseDevice(&dev, &prop)); // Choose a device with compute capability >= 3.0
	cudaCheckErrors(cudaGLSetGLDevice(dev));

	return dev;
}

void InitCudaTexture(cudaGraphicsResource_t& textureResource, cudaResourceDesc& resourceDesc, uint32_t textureID)
{
	// Register texture with CUDA resource
	cudaCheckErrors(cudaGraphicsGLRegisterImage(&textureResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
}
