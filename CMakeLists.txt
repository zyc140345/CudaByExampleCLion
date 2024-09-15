cmake_minimum_required(VERSION 3.22.1)
project(cuda_by_example)

set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS})

include_directories(appendix_a)
include_directories(common)
include_directories(common/GL)

add_subdirectory(chapter03)
add_subdirectory(chapter04)
add_subdirectory(chapter05)
add_subdirectory(chapter06)
add_subdirectory(chapter07)
add_subdirectory(chapter08)
add_subdirectory(chapter09)
add_subdirectory(chapter10)
add_subdirectory(chapter11)
add_subdirectory(appendix_a)