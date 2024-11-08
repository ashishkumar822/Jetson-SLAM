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

#define CUDA_NUM_THREADS_PER_BLOCK_x 32
#define CUDA_NUM_THREADS_PER_BLOCK_y 8

#define GRID_LAUNCH


__global__ void imresize_GPU_pitched(int n_threads,
                                     int height, int width,
                                     int op_height, int op_width,
                                     float inv_scale,
                                     unsigned char* ip_image,int ip_pitch,
                                     unsigned char* op_image,int op_pitch)
{
#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if(w < op_width && h < op_height)
    {
        const int offset = h * op_pitch + w;

#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int h = index / op_width;
        int w = index % op_width;

        const int offset = h * op_pitch + w;

#endif
        float scale = 1.0f / inv_scale;

        float ip_h = scale * h;
        float ip_w = scale * w;

        int xl = floorf(ip_w);
        int xr = xl + 1;

        int yt = floorf(ip_h);
        int yb = yt + 1;

        float w_xl = xr - ip_w;
        float w_xr = 1  - w_xl;
        float w_yt = yb - ip_h;
        float w_yb = 1  - w_yt;

        op_image[offset] = w_yt * w_xl * ip_image[yt * ip_pitch + xl]
                + w_yt * w_xr * ip_image[yt * ip_pitch + xr]
                + w_yb * w_xl * ip_image[yb * ip_pitch + xl]
                + w_yb * w_xr * ip_image[yb * ip_pitch + xr];

    }

}



void ORB_GPU::Compute_pyramid(int height, int width,
                              int op_height, int op_width,
                              float inv_scale,
                              unsigned char* ip_image_data_gpu,
                              int ip_pitch,
                              unsigned char* op_image_data_gpu,
                              int op_pitch,
                              cudaStream_t& cuda_stream)
{

    int n_threads =  op_height * op_width;

#ifdef GRID_LAUNCH

    int CUDA_NUM_BLOCKS_x = (op_width  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
    int CUDA_NUM_BLOCKS_y = (op_height - 1) / CUDA_NUM_THREADS_PER_BLOCK_y + 1;

    dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
    dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, CUDA_NUM_THREADS_PER_BLOCK_y, 1);

    imresize_GPU_pitched<<<grid_dim, block_dim, 0, cuda_stream>>>(
                                                                    n_threads,
                                                                    height, width,
                                                                    op_height, op_width,
                                                                    inv_scale,
                                                                    ip_image_data_gpu, ip_pitch,
                                                                    op_image_data_gpu, op_pitch);
#else
    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;


    imresize_GPU_pitched<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                            n_threads,
                                                                                            height, width,
                                                                                            op_height, op_width,
                                                                                            inv_scale,
                                                                                            ip_image_data_gpu, ip_pitch,
                                                                                            op_image_data_gpu, op_pitch);

#endif

}





__global__ void imresize_GPU(int n_threads,
                             int height, int width,
                             int op_height, int op_width,
                             float inv_scale,
                             unsigned char* ip_image,
                             unsigned char* op_image)
{
    int index = blockDim.x * blockIdx.x +  threadIdx.x;

    if(index < n_threads)
    {
        int h = index / op_width;
        int w = index % op_width;

        float scale = 1.0f / inv_scale;

        float ip_h = scale * h;
        float ip_w = scale * w;

        int xl = floorf(ip_w);
        int xr = xl + 1;

        int yt = floorf(ip_h);
        int yb = yt + 1;

        float w_xl = xr - ip_w;
        float w_xr = 1  - w_xl;
        float w_yt = yb - ip_h;
        float w_yb = 1  - w_yt;

        op_image[index] = w_yt * w_xl * ip_image[yt * width + xl]
                + w_yt * w_xr * ip_image[yt * width + xr]
                + w_yb * w_xl * ip_image[yb * width + xl]
                + w_yb * w_xr * ip_image[yb * width + xr];

    }

}



void ORB_GPU::Compute_pyramid(int height, int width,
                              int op_height, int op_width,
                              float inv_scale,
                              unsigned char* ip_image_data_gpu,
                              unsigned char* op_image_data_gpu,
                              cudaStream_t& cuda_stream)
{
    int n_threads =  op_height * op_width;

    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;


    imresize_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                    n_threads,
                                                                                    height, width,
                                                                                    op_height, op_width,
                                                                                    inv_scale,
                                                                                    ip_image_data_gpu,
                                                                                    op_image_data_gpu);
}




__global__ void dummy_GPU()
{

}

// without this first time kernel takes roughly 2uS more
void ORB_GPU::dummy_kernel_launch_tosetup_context()
{
    int n_threads =  512 * 512;

    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

    for(int i=0;i<n_levels_;i++)
        dummy_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_streams_[i] >>>();

    for(int i=0;i<n_levels_;i++)
        cudaStreamSynchronize(cuda_streams_[i]);
}

}
