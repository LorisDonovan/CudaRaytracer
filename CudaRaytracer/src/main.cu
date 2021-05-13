#include <iostream>
#include <string>
#include <vector>
#include <cassert>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include "vec3.h"
#include "ray.h"

#define cudaCheckErrors(val) CheckCuda(val, #val, __FILE__, __LINE__)


// Window property
constexpr float aspectRatio = 16.0f / 9.0f;
const uint32_t height = 540;
const uint32_t width  = static_cast<uint32_t>(height * aspectRatio);

// CUDA resources
cudaResourceDesc       resourceDesc;
cudaGraphicsResource_t textureResource;
cudaArray_t            textureArray;
cudaSurfaceObject_t    surfaceObj = 0;


// ----------GLFW------------------------------------------
GLFWwindow* InitWindow();

// ----------OpenGL----------------------------------------
uint32_t CreateShader(const char* vertexShader, const char* fragShader);
uint32_t CompileShader(uint32_t type, const char* shaderSrc);
void ShaderBind(uint32_t shaderID);
void ShaderUnbind();
void ShaderSetInt(uint32_t shaderID, const std::string& name, int32_t value);
void InitFbQuad(uint32_t& quadVA, uint32_t& quadVB, uint32_t& shaderID);
uint32_t InitGLTexture();
void TextureBind(uint32_t textureID, uint32_t slot = 0);
void Cleanup(uint32_t& quadVao, uint32_t& quadVbo, uint32_t& textureID, uint32_t& shaderID);

// ----------CUDA------------------------------------------
void CheckCuda(cudaError_t result, const char* func, const char* filepath, const uint32_t line);
int32_t InitCudaDevice();
void InitCudaTexture(uint32_t textureID);

// ---------Raytracer--------------------------------------
__global__ void Kernel(cudaSurfaceObject_t surfaceObj, vec3 origin, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical);
__device__ vec3 RayColor(const Ray& ray);
__device__ bool HitSphere(const vec3& center, float radius, const Ray& ray);


int main(int argc, char** argv)
{
	// Camera and viewport
	float focalLength    = 1.0f;
	float viewportHeight = 2.0f;
	float viewportWidth  = aspectRatio * viewportHeight;
	vec3  origin(0.0f, 0.0f, 0.0f);
	vec3  horizontal(viewportWidth, 0.0f, 0.0f);
	vec3  vertical(0.0f, viewportHeight, 0.0f);
	vec3  lowerLeftCorner = origin - horizontal * 0.5f - vertical * 0.5f - vec3(0.0f, 0.0f, focalLength);

	// Initialize opengl and cuda interop
	GLFWwindow* window = InitWindow();
	int32_t cudaDevID  = InitCudaDevice();

	// Initialize vertex array and vertex buffer
	uint32_t quadVA, quadVB, shaderID;
	InitFbQuad(quadVA, quadVB, shaderID);

	// Initialize texture
	uint32_t textureID = InitGLTexture();
	InitCudaTexture(textureID);

	// CUDA kernel thread layout
	int32_t numThreads = 32;
	dim3 blocks((width + numThreads - 1) / numThreads, (height + numThreads - 1) / numThreads);
	dim3 threads(numThreads, numThreads);

	// CUDA register and create surface object resource
	cudaCheckErrors(cudaGraphicsMapResources(1, &textureResource));
	cudaCheckErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0));
	resourceDesc.res.array.array = textureArray;
	cudaCheckErrors(cudaCreateSurfaceObject(&surfaceObj, &resourceDesc));
	Kernel<<<blocks, threads>>>(surfaceObj, origin, lowerLeftCorner, horizontal, vertical);
	cudaCheckErrors(cudaGraphicsUnmapResources(1, &textureResource)); // sync cuda operations before graphics calls

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

	Cleanup(quadVA, quadVB, textureID, shaderID);
	glfwTerminate();
	return 0;
}


// ----------GLFW------------------------------------------
GLFWwindow* InitWindow() 
{
	// Initialize and configure glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a windowed mode window and its OpenGL context
	GLFWwindow* window = glfwCreateWindow(width, height, "CudaRaytracer", nullptr, nullptr);
	if (!window)
	{
		std::cerr << "ERROR: Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(-1);
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Initialize glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cerr << "ERROR: Failed to initialize GLAD" << std::endl;
		exit(-1);
	}

	std::cout << "OpenGL info:\n";
	std::cout << "    Vendor  : " << glGetString(GL_VENDOR)   << "\n";
	std::cout << "    Renderer: " << glGetString(GL_RENDERER) << "\n";
	std::cout << "    Version : " << glGetString(GL_VERSION)  << std::endl;

	return window;
}


// ----------OpenGL----------------------------------------
uint32_t CreateShader(const char* vertexShader, const char* fragShader)
{
	uint32_t program = glCreateProgram();
	uint32_t vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
	uint32_t fs = CompileShader(GL_FRAGMENT_SHADER, fragShader);

	// linking
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glValidateProgram(program);

	// delete the intermediates
	glDeleteShader(vs);
	glDeleteShader(fs);

	return program;
}

uint32_t CompileShader(uint32_t type, const char* shaderSrc)
{
	uint32_t id = glCreateShader(type);
	glShaderSource(id, 1, &shaderSrc, nullptr);
	glCompileShader(id);

	// Error Handling
	int32_t result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &maxLength);

		std::vector<GLchar> infoLog(maxLength);
		glGetShaderInfoLog(id, maxLength, &maxLength, &infoLog[0]);

		glDeleteShader(id);

		if (type == GL_VERTEX_SHADER)
			std::cerr << "Failed To Compile Vertex Shader!" << std::endl;
		if (type == GL_FRAGMENT_SHADER)
			std::cerr << "Failed To Compile Fragment Shader!" << std::endl;
		std::cout << infoLog.data() << std::endl;
		assert(false);
	}

	return id;
}

void ShaderBind(uint32_t shaderID)
{
	glUseProgram(shaderID);
}

void ShaderUnbind()
{
	glUseProgram(0);
}

void ShaderSetInt(uint32_t shaderID, const std::string& name, int32_t value)
{
	GLint location = glGetUniformLocation(shaderID, name.c_str());
	glUniform1i(location, value);
}

void InitFbQuad(uint32_t& quadVA, uint32_t& quadVB, uint32_t& shaderID)
{
	float quadVertices[] = {
		// positions   // texCoords
		-1.0f,  1.0f,  0.0f, 1.0f,
		-1.0f, -1.0f,  0.0f, 0.0f,
		 1.0f, -1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f,  0.0f, 1.0f,
		 1.0f, -1.0f,  1.0f, 0.0f,
		 1.0f,  1.0f,  1.0f, 1.0f
	};

	glGenBuffers(1, &quadVB);
	glGenVertexArrays(1, &quadVA);

	glBindVertexArray(quadVA);
	glBindBuffer(GL_ARRAY_BUFFER, quadVB);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	// Position Attribute
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// Texture coordinate Attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// Shader
	const char* vertexShader = R"(
		#version 450 core

		layout(location = 0) in vec2 a_Position;
		layout(location = 1) in vec2 a_TexCoords;

		out vec2 TexCoords;

		void main()
		{
			TexCoords   = a_TexCoords;
			gl_Position = vec4(a_Position, 0.0f, 1.0f);
		}
	)";

	const char* fragShader = R"(
		#version 450 core

		in  vec2 TexCoords;
		out vec4 FragColor;

		uniform sampler2D u_Texture;

		void main()
		{
			vec3 col  = texture(u_Texture, TexCoords).rgb; 
			FragColor = vec4(col, 1.0f);
			//FragColor = vec4(TexCoords, 0.0f, 1.0f);
		}
	)";

	shaderID = CreateShader(vertexShader, fragShader);
	ShaderBind(shaderID);
	ShaderSetInt(shaderID, "u_Texture", 0);
	ShaderUnbind();
}

uint32_t InitGLTexture()
{
	uint32_t textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	// Texture properties
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// set texture image
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	// Unbind texture
	glBindTexture(GL_TEXTURE_2D, 0);

	return textureID;
}

void TextureBind(uint32_t textureID, uint32_t slot)
{
	glActiveTexture(GL_TEXTURE0 + slot);
	glBindTexture(GL_TEXTURE_2D, textureID);
}

void Cleanup(uint32_t& quadVao, uint32_t& quadVbo, uint32_t& textureID, uint32_t& shaderID)
{
	glDeleteBuffers(1, &quadVbo);
	glDeleteVertexArrays(1, &quadVao);
	glDeleteTextures(1, &textureID); 
	glDeleteProgram(shaderID);
}


// ----------CUDA------------------------------------------
void CheckCuda(cudaError_t result, const char* func, const char* filepath, const uint32_t line)
{
	if (result)
	{
		std::cerr << "CUDA::ERROR:" << static_cast<uint32_t>(result) << " in file: \"" << filepath << "\": line " << line << " - '" << func << "'" << std::endl;
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

void InitCudaTexture(uint32_t textureID)
{
	// Register texture with CUDA resource
	cudaCheckErrors(cudaGraphicsGLRegisterImage(&textureResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
}


// ---------Raytracer--------------------------------------
__global__ void Kernel(cudaSurfaceObject_t surfaceObj, vec3 origin, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if ((x >= width) || (y >= height))
		return;

	// Offset values to move the ray across the screen
	float u    = float(x) / float(width);
	float v    = float(y) / float(height);
	// Calculate color
	vec3 color = RayColor(Ray(origin, lowerLeftCorner + u * horizontal + v * vertical - origin));
	uint8_t r  = uint8_t(color.r() * 255);
	uint8_t g  = uint8_t(color.g() * 255);
	uint8_t b  = uint8_t(color.b() * 255);

	uchar4 data = make_uchar4(r, g, b, 255);
	surf2Dwrite(data, surfaceObj, x * sizeof(uchar4), y);
}

__device__ vec3 RayColor(const Ray& ray)
{
	if (HitSphere(vec3(0.0f, 0.0f, -2.0f), 1.0f, ray))
		return vec3(1.0f, 0.0f, 0.0f);

	vec3 dir = ray.GetDirection();      // Direction of ray is a unit vector
	float t  = 0.5f * (dir.y() + 1.0f); // Mapping y in the range [0, 1]
	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f); // Blend the background from blue to white vertically
}

__device__ bool HitSphere(const vec3& center, float radius, const Ray& ray)
{
	vec3 oc = ray.GetOrigin() - center;
	float a = Dot(ray.GetDirection(), ray.GetDirection());
	float b = 2.0f * Dot(ray.GetDirection(), oc);
	float c = Dot(oc, oc) - radius * radius;
	float dis = b * b - 4.0f * a * c; // discriminant

	return (dis > 0.0f);
}

