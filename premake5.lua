workspace "CudaRaytracer"
	architecture "x64"
	startproject "CudaRaytracer"
	
	configurations{
		"Debug",
		"Release"
	}

	flags {
		"MultiprocessorCompile"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "CudaRaytracer"
	location "CudaRaytracer"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "off"

	targetdir ("bin/" ..outputdir.. "/%{prj.name}")
	objdir("bin-int/" ..outputdir.. "/%{prj.name}")

	files {
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.c",
		"%{prj.name}/src/**.cu",
		"%{prj.name}/src/**.cpp"
	}

	includedirs {
		"%{prj.name}/src",
		"ext/glad/include",
		"ext/glfw/include"
	}

	libdirs {
		"ext/glfw/lib"
	}

	links {
		"cudart_static.lib",
		"cublas.lib",
		"curand.lib",
		"opengl32.lib",
		"glfw3.lib"
	}

	filter "system:windows"
		systemversion "latest"

		filter "configurations:Debug"
			runtime "Debug"
			symbols "on"

		filter "configurations:Release"
			runtime "Release"
			optimize "on"

