#include "windowInit.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>


GLFWwindow* InitWindow(uint32_t width, uint32_t height)
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
	std::cout << "    Vendor  : " << glGetString(GL_VENDOR) << "\n";
	std::cout << "    Renderer: " << glGetString(GL_RENDERER) << "\n";
	std::cout << "    Version : " << glGetString(GL_VERSION) << std::endl;

	return window;
}
