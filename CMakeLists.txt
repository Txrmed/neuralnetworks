include(FetchContent)

cmake_minimum_required(VERSION 4.1)
project(neuralnetworks)

set(CMAKE_CXX_STANDARD 23)

# Install the Eigen library
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.0
)

set(SOURCES src/main.cpp src/Network.cpp src/Dataset.cpp)

add_executable(neuralnetworks ${SOURCES})

target_include_directories(neuralnetworks PRIVATE include)

FetchContent_MakeAvailable(Eigen)
target_link_libraries(neuralnetworks PRIVATE Eigen3::Eigen)

