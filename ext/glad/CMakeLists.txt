cmake_minimum_required(VERSION 3.22)

project(glad LANGUAGES C)

add_library(
	${PROJECT_NAME}
	glad.c
)

target_include_directories(
	${PROJECT_NAME}
	PUBLIC
	"include/"
)
