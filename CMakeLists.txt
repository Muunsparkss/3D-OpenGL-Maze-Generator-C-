cmake_minimum_required(VERSION 3.16)

# Enable BOTH C and C++
project(opengl_maze LANGUAGES C CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Executable (note glad.c is C) ----
add_executable(opengl_maze
    src/main.cpp
    src/glad.c          # GLAD 1 implementation
)

# GLAD headers live next to main.cpp (glad.h + KHR/khrplatform.h)
target_include_directories(opengl_maze PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- Dependencies via vcpkg (CONFIG mode) ----
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(opengl_maze PRIVATE
    glfw
    glm::glm
)

# ---- Platform GL lib ----
if (WIN32)
    target_link_libraries(opengl_maze PRIVATE opengl32)
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(opengl_maze PRIVATE OpenGL::GL)
endif()

# ---- Warnings ----
if (MSVC)
    target_compile_definitions(opengl_maze PRIVATE NOMINMAX)
    target_compile_options(opengl_maze PRIVATE /W4 /permissive-)
else()
    target_compile_options(opengl_maze PRIVATE -Wall -Wextra -Wpedantic)
endif()
