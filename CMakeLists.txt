cmake_minimum_required(VERSION 3.24)

project(CudaRaytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 60) # set the card's architecture level (compute capability)

# set Visual Studio startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

file(GLOB_RECURSE CUDA_RAYTRACER_CU_FILES "${PROJECT_SOURCE_DIR}/CudaRaytracer/src/*.cu")
file(GLOB_RECURSE CUDA_RAYTRACER_H_FILES "${PROJECT_SOURCE_DIR}/CudaRaytracer/src/*.h")
add_executable(
	${PROJECT_NAME}
	"${CUDA_RAYTRACER_CU_FILES}"
	"${CUDA_RAYTRACER_H_FILES}"
)

add_subdirectory("ext/glad")

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# We need to explicitly state that we need all CUDA files in the
# particle library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(
	${PROJECT_NAME}
	PUBLIC
	"${CUDAToolkit_INCLUDE_DIRS}"
	"${PROJECT_SOURCE_DIR}/CudaRaytracer/src/"
	"${PROJECT_SOURCE_DIR}/ext/"
	"${PROJECT_SOURCE_DIR}/ext/glad/include/"
	"${PROJECT_SOURCE_DIR}/ext/glfw/include/"
)

target_link_libraries(
	${PROJECT_NAME}
	"cudart_static.lib"
	"cublas.lib"
	"curand.lib"
	"${OPENGL_gl_LIBRARY}"
	"${PROJECT_SOURCE_DIR}/ext/glfw/lib/glfw3.lib"
	glad
)
