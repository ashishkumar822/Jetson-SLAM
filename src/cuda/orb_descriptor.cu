/**
* This file is part of Jetson-SLAM.
*
* Written by Ashish Kumar Indian Institute of Tehcnology, Kanpur, India
* For more information see <https://github.com/ashishkumar822/Jetson-SLAM>
*
* Jetson-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Jetson-SLAM is distributed WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
*/



#include "cuda/orb_gpu.hpp"

#include<chrono>


namespace orb_cuda {


#define CUDA_NUM_THREADS_PER_BLOCK 512


__global__ void ORB_compute_descriptorGPU(int n_threads,
                                          int height, int width,
                                          unsigned char* image_data,
                                          int image_pitch,
                                          signed char* pattern_x_data,
                                          signed char* pattern_y_data,
                                          int n_keypoints,
                                          int* keypoints_x,
                                          int* keypoints_y,
                                          float* keypoints_angle,
                                          unsigned char* keypoints_descriptor)
{
    int index = blockDim.x * blockIdx.x +  threadIdx.x;

    if(index < n_threads)
    {
        int h = index / 32;
        int w = index % 32;

        float angle = keypoints_angle[h];
        float a = cosf(angle);
        float b = sinf(angle);

        const unsigned char* center = image_data + keypoints_y[h] * image_pitch + keypoints_x[h];
        const signed char* pattern_x = pattern_x_data + w * 16;
        const signed char* pattern_y = pattern_y_data + w * 16;

        unsigned char val = 0x00;

        for(int i=0;i<8;i++)
        {
            int t0;
            int t1;

            int first_idx = i*2;
            int second_idx = i*2+1;
            {
                int patternx = pattern_x[first_idx];
                int patterny = pattern_y[first_idx];
                int rowoffset = rintf(patternx * b + patterny * a) * image_pitch;
                int coloffset = rintf(patternx * a - patterny * b);

                t0 = center[rowoffset + coloffset];
            }
            {
                int patternx = pattern_x[second_idx];
                int patterny = pattern_y[second_idx];
                int rowoffset = rintf(patternx * b + patterny * a) * image_pitch;
                int coloffset = rintf(patternx * a - patterny * b);

                t1 = center[rowoffset + coloffset];
            }
            val |= ((t0 < t1) << i);
        }
       keypoints_descriptor[h * 32 + w] = val;
    }

}



void ORB_GPU::ORB_compute_descriptor(int height, int width,
                                     unsigned char* image_data_gpu,
                                     int image_pitch,
                                     signed char* pattern_x_gpu,
                                     signed char* pattern_y_gpu,
                                     int n_keypoints,
                                     int* keypoints_x_gpu,
                                     int* keypoints_y_gpu,
                                     float* keypoints_angle_gpu,
                                     unsigned char* keypoints_descriptor_gpu,
                                     cudaStream_t& cuda_stream)
{
    {
        int n_threads =  n_keypoints * 32;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK  + CUDA_NUM_THREADS_PER_BLOCK;


        ORB_compute_descriptorGPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                     n_threads,
                                                                                                     height, width,
                                                                                                     image_data_gpu,
                                                                                                     image_pitch,
                                                                                                     pattern_x_gpu,
                                                                                                     pattern_y_gpu,
                                                                                                     n_keypoints,
                                                                                                     keypoints_x_gpu,
                                                                                                     keypoints_y_gpu,
                                                                                                     keypoints_angle_gpu,
                                                                                                     keypoints_descriptor_gpu
                                                                                                     );

    }
}

}
