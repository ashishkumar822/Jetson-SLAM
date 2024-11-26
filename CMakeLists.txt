cmake_minimum_required(VERSION 2.8)
project(Jetson-SLAM)



IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wno-error -Wno-sign-compare -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-sign-compare  -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

#if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   #add_definitions(-DCOMPILEDWITHC14)
   message(STATUS "Using flag -std=c++14.")
#elseif(COMPILER_SUPPORTS_CXX11)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#   add_definitions(-DCOMPILEDWITHC11)
#   message(STATUS "Using flag -std=c++11.")
#elseif(COMPILER_SUPPORTS_CXX0X)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#   add_definitions(-DCOMPILEDWITHC0X)
#   message(STATUS "Using flag -std=c++0x.")
#else()
#   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif()

message(STATUS "${CMAKE_CXX_FLAGS}")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "--std c++14")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4 REQUIRED)

if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV >= 4 not found.")
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${PROJECT_SOURCE_DIR}/Thirdparty/
${PROJECT_SOURCE_DIR}/Thirdparty/g2o
${Pangolin_INCLUDE_DIRS}
${CUDA_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)




cuda_add_library(${PROJECT_NAME} SHARED
src/Converter.cpp
src/Frame.cpp
src/FrameDrawer.cpp
src/Initializer.cpp
src/KeyFrame.cpp
src/KeyFrameDatabase.cpp
src/LocalMapping.cpp
src/LoopClosing.cpp
src/Map.cpp
src/MapDrawer.cpp
src/MapPoint.cpp
src/Optimizer.cpp
src/ORBextractor.cpp
src/ORBmatcher.cpp
src/PnPsolver.cpp
src/Sim3Solver.cpp
src/System.cpp
src/Tracking.cpp
src/Viewer.cpp
src/cuda/orb_bitpattern.cpp
src/cuda/orb_copy_output.cu
src/cuda/orb_descriptor.cu
src/cuda/orb_FAST_apply_NMS_G.cu
src/cuda/orb_FAST_apply_NMS_L.cu
src/cuda/orb_FAST_apply_NMS_MS.cpp
src/cuda/orb_FAST_apply_NMS_MS.cu
src/cuda/orb_FAST_compute_score.cu
src/cuda/orb_FAST_obtain_keypoints.cpp
src/cuda/orb_FAST_orientation.cu
src/cuda/orb_gaussian.cu
src/cuda/orb_gpu.cpp
src/cuda/orb_matcher.cu
src/cuda/orb_pyramid.cu
src/cuda/orb_stereo_match.cu
src/cuda/synced_mem_holder.cpp
src/cuda/tracking_isinfrustum.cu
)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${CUDA_LIBRARIES}
${CUDA_CUBLAS_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

# Build execs

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti Examples/Stereo/stereo_kitti.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc Examples/Stereo/stereo_euroc.cpp)
target_link_libraries(stereo_euroc ${PROJECT_NAME})


add_executable(stereo_live Examples/Stereo/stereo_live.cpp)
target_link_libraries(stereo_live ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cpp)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

add_executable(rgbd_live
Examples/RGB-D/rgbd_live.cpp)
target_link_libraries(rgbd_live ${PROJECT_NAME})



set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_live
Examples/Monocular/mono_live.cpp)
target_link_libraries(mono_live ${PROJECT_NAME})


add_executable(mono_tum
Examples/Monocular/mono_tum.cpp)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cpp)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
Examples/Monocular/mono_euroc.cpp)
target_link_libraries(mono_euroc ${PROJECT_NAME})


