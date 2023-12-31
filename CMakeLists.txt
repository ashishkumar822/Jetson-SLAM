cmake_minimum_required(VERSION 2.8)
project(Jetson-SLAM)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(CUDA REQUIRED)


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV >= 3 not found.")
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${CUDA_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


cuda_add_library(${PROJECT_NAME} SHARED
src/Converter.cc
src/Frame.cc
src/FrameDrawer.cc
src/Initializer.cc
src/KeyFrame.cc
src/KeyFrameDatabase.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/Map.cc
src/MapDrawer.cc
src/MapPoint.cc
src/Optimizer.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/PnPsolver.cc
src/Sim3Solver.cc
src/System.cc
src/Tracking.cc
src/Viewer.cc
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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/execs)

add_executable(stereo_kitti execs/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc execs/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_kaistvio execs/stereo_kaistvio.cc)
target_link_libraries(stereo_kaistvio ${PROJECT_NAME})

add_executable(stereo_live execs/stereo_live.cpp)
target_link_libraries(stereo_live ${PROJECT_NAME})


