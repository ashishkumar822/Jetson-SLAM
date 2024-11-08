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
#include<tictoc.hpp>

using namespace std;

namespace orb_cuda {

#define CUDA_NUM_THREADS_PER_BLOCK 32


#define CUDA_NUM_THREADS_PER_BLOCK_x 32
#define CUDA_NUM_THREADS_PER_BLOCK_y 5


#define GRID_LAUNCH

#define NMS_WINDOW 3
#define NMS_WINDOW_HALF (NMS_WINDOW - 1 ) / 2

#define DIVERGENCE_LESS_CODE




__global__ void Tile_unrolling_reduction_kernel_v2(int imheight, int imwidth,
                                                   int tile_h, int tile_w,
                                                   int n_tiles_h, int n_tiles_w,
                                                   int n_loc_per_thread,
                                                   int n_threads_y_per_tile,
                                                   int n_tiles_per_block,
                                                   int* score_data_gpu,
                                                   int score_pitch,
                                                   int* kp_x,
                                                   int* kp_y,
                                                   int* kp_score,
                                                   int fuse_nms_L_with_nms_G)
{

    const int n_threads_per_tile = n_threads_y_per_tile;

#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    int block_max = n_tiles_per_block * tile_w;


    int h_im = (h/n_threads_per_tile) * tile_h;
    const int w_im =  blockIdx.x * block_max + threadIdx.x;

    int tile_boundry_h_min = h_im;
    int tile_boundry_h_max = h_im + tile_h;

    if(tile_boundry_h_min < BORDER_SKIP)
        tile_boundry_h_min = BORDER_SKIP;

    if(tile_boundry_h_max > imheight - BORDER_SKIP)
        tile_boundry_h_max = imheight - BORDER_SKIP;


    __shared__ int shared_score[128*10];
    __shared__ int shared_x[128*10];
    __shared__ int shared_y[128*10];



    int max_score = 0;
    int max_x = w_im;
    int max_y = h_im;

    int my_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int my_idx_reduce  = threadIdx.y;


    if(w_im < imwidth && threadIdx.x < block_max)
    {

#else
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int h = thread_idx / roiwidth;
        int w = thread_idx % roiwidth;
#endif

        //vertical cell aggregation


        const int mini_tile = (tile_h - 1) / n_threads_per_tile + 1;

        const int h_start = h_im + my_idx_reduce;

        for(int i=0;i<mini_tile;i++)
        {
            int h = h_start + i * n_threads_per_tile;

            if(h >=tile_boundry_h_min && h < tile_boundry_h_max)
            {
                int score = score_data_gpu[h * score_pitch + w_im];

                if(fuse_nms_L_with_nms_G)
                {
                    int valid = 1;
                    //unrolled for 3x3
                    int row_offset = (h + -1) * score_pitch;
                    int col_offset = w_im + -1;
                    int total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    row_offset = (h + 0) * score_pitch;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    //                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];


                    row_offset = (h + 1) * score_pitch;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    score *= valid;
                }

                if(score > max_score)
                {
                    max_score = score;
                    max_y = h;
                }
            }
        }
    }

    shared_score[my_idx] = max_score;
    shared_y[my_idx] = max_y;

    // shouldn't be called inside only if blocks
    __syncthreads();

    int tile_idx_w = threadIdx.x / tile_w;

    if(w_im < imwidth && threadIdx.x < block_max)
    {
        if(my_idx_reduce == 0)
        {
            for(int i=1;i<n_threads_per_tile;i++)
            {
                int offset = i*blockDim.x+threadIdx.x;
                int temp_score = shared_score[offset];

                if(max_score < temp_score)
                {
                    max_score = temp_score;
                    max_y = shared_y[offset];
                }

            }

            shared_score[my_idx] = max_score;
            shared_x[my_idx] = max_x;
            shared_y[my_idx] = max_y;
        }
    }

    __syncthreads();


    //horizontal cell aggregation
    //log2
    int tile_loc_w = threadIdx.x % tile_w;
    int log2_tile_w = ceilf(log2f((float)tile_w));

    int group_size = (tile_w - 1) / 2 + 1;

    for(int i=0;i<log2_tile_w;i++)
    {
        if(w_im < imwidth && my_idx_reduce == 0 && tile_loc_w < group_size && threadIdx.x < block_max)
        {
            int offset = tile_loc_w +  group_size;

            if(offset < tile_w)
            {
                offset = my_idx + group_size;

                int temp_score = shared_score[offset];

                if(max_score < temp_score)
                {
                    max_score = temp_score;
                    max_y = shared_y[offset];
                    max_x = shared_x[offset];
                }
            }

            shared_score[my_idx] = max_score;
            shared_x[my_idx] = max_x;
            shared_y[my_idx] = max_y;
        }

        group_size = (group_size - 1) / 2 + 1;

        __syncthreads();
    }

    //MLPT
    {

    }


    if(w_im < imwidth && my_idx_reduce == 0 && tile_loc_w == 0 && threadIdx.x < block_max)
    {

        int tile_idx_h_img = blockIdx.y;
        int tile_idx_w_img = blockIdx.x * n_tiles_per_block + tile_idx_w;

        int tile_idx = tile_idx_h_img * n_tiles_w + tile_idx_w_img;

        kp_score[tile_idx] = max_score;
        kp_x[tile_idx] = max_x;
        kp_y[tile_idx] = max_y;

    }
}


void ORB_GPU::FAST_apply_NMS_G_reduce_unroll_reduce(int height, int width,
                                                    int tile_h, int tile_w,
                                                    int n_tiles_h, int n_tiles_w,
                                                    int warp_tile_h, int warp_tile_w,
                                                    int fuse_nms_L_with_nms_G,
                                                    int* image_unroll_gpu,
                                                    int* image_unroll_x_gpu,
                                                    int* image_unroll_y_gpu,
                                                    int* score_data_gpu,
                                                    int score_pitch,
                                                    int* keypoints_x,
                                                    int* keypoints_y,
                                                    int* keypoints_score,
                                                    cudaStream_t& cuda_stream)
{

    {
        int n_loc_per_thread = std::max(1, std::min(10, tile_w/3));

        std::vector<int> block_sizes_x = {128};

        int residual = 1E5;
        int block_size_x = -1;

        for(int i=0;i<block_sizes_x.size();i++)
        {
            int tmp_block_size_x = block_sizes_x[i];

            int tmp_residual = tmp_block_size_x % tile_w;

            if(tmp_residual < residual)
            {
                residual = tmp_residual;
                block_size_x = tmp_block_size_x;
            }
        }

        if(n_loc_per_thread > tile_h)
            n_loc_per_thread = tile_h;

        int n_threads_y_per_tile = (tile_h - 1) / n_loc_per_thread + 1;

        if(n_threads_y_per_tile * block_size_x > 1024)
            n_threads_y_per_tile = 1024 / block_size_x;


        int n_tiles_per_block = block_size_x / tile_w;

        //shared_mem_size


        int CUDA_NUM_BLOCKS_x = (n_tiles_w  - 1) / n_tiles_per_block + 1;
        int CUDA_NUM_BLOCKS_y = n_tiles_h;

//        std::cout << CUDA_NUM_BLOCKS_x << " "  << CUDA_NUM_BLOCKS_y << "\n";
//        std::cout << block_size_x << " "  << n_threads_y_per_tile << "\n";

        dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
        dim3 block_dim(block_size_x, n_threads_y_per_tile, 1);


        int roiheight = (height - 2*0);
        int roiwidth  = (width - 2*0);

        Tile_unrolling_reduction_kernel_v2<<<grid_dim, block_dim, 0, cuda_stream>>>(
                                                                                      height, width,
                                                                                      tile_h, tile_w,
                                                                                      n_tiles_h, n_tiles_w,
                                                                                      n_loc_per_thread,
                                                                                      n_threads_y_per_tile,
                                                                                      n_tiles_per_block,
                                                                                      score_data_gpu,
                                                                                      score_pitch,
                                                                                      keypoints_x,
                                                                                      keypoints_y,
                                                                                      keypoints_score,
                                                                                      fuse_nms_L_with_nms_G);

        return;
    }
}


}

