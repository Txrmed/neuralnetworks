include(FetchContent)

cmake_minimum_required(VERSION 4.1)
project(digitrecognition)

set(CMAKE_CXX_STANDARD 23)

# Install the Eigen library
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.0
)

set(SOURCES src/main.cpp src/Network.cpp src/Dataset.cpp)

add_executable(digitrecognition ${SOURCES})

target_include_directories(digitrecognition PRIVATE include)

FetchContent_MakeAvailable(Eigen)
target_link_libraries(digitrecognition PRIVATE Eigen3::Eigen)

