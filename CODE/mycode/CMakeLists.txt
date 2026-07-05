cmake_minimum_required(VERSION 3.16)

project(JacobiCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 86)

find_package(OpenMP REQUIRED)

set(JS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../JacobiSetComputation-master")
set(TRIMESH_PATH "${JS_DIR}/trimesh2" CACHE PATH "Path to Trimesh library")
set(TRIMESH_LIB "${TRIMESH_PATH}/lib.Linux64/libtrimesh.a" CACHE FILEPATH "Trimesh library")
set(SOS_PATH "${JS_DIR}/Detri_2.6.a" CACHE PATH "Path to SOS library")
set(SOS_LIB "${SOS_PATH}/build/lib/libSoS.a" CACHE FILEPATH "SOS library")

add_executable(jacobi_cuda
    main_cuda.cpp
    jacobi_cuda.cu
    "${JS_DIR}/src/TriMeshJ.cpp"
    "${JS_DIR}/src/JacobiSet.cpp"
)

target_include_directories(jacobi_cuda PRIVATE
    "${JS_DIR}/include"
    "${TRIMESH_PATH}/include"
    "${SOS_PATH}/basic"
    "${SOS_PATH}/lia"
    "${SOS_PATH}/sos"
)

target_compile_definitions(jacobi_cuda PRIVATE USE_SOS)

target_link_libraries(jacobi_cuda PRIVATE
    "${TRIMESH_LIB}"
    "${SOS_LIB}"
    OpenMP::OpenMP_CXX
)
