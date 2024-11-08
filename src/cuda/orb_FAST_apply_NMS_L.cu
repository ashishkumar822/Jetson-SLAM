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

#define NMS_WINDOW 3
#define NMS_WINDOW_HALF (NMS_WINDOW - 1 ) / 2


#define CUDA_NUM_THREADS_PER_BLOCK_x 32 // to ensure memory coalescing 16x16 can be an option however half of the warp will access different mem while anlther half the other
#define CUDA_NUM_THREADS_PER_BLOCK_y 8


#define GRID_LAUNCH


__global__ void FASTapplyNMSGPU(int n_threads,
                                int imheight, int imwidth,
                                int roiheight, int roiwidth,
                                int* score_data_gpu,
                                int score_pitch,
                                int* score_data_nms,
                                int score_nms_pitch)
{
#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if(w < imwidth && h < imheight)
    {
#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int h = index / roiwidth;
        int w = index % roiwidth;
#endif

        int MAX_BORDER = max(BORDER_SKIP, NMS_WINDOW_HALF);

        if(w >= MAX_BORDER && w < imwidth - MAX_BORDER && h < imheight - MAX_BORDER && h >= MAX_BORDER)
        {


            const int h_im = h;
            const int w_im = w;

            int score = score_data_gpu[h_im * score_pitch + w_im];

            int tile_h = 31;
            int tile_w = 31;

            int tile_loc_h = h % tile_h;
            int tile_loc_w = w % tile_w;

            if(tile_loc_h == 0 || tile_loc_h == tile_h - 1 || tile_loc_w == 0 || tile_loc_w == tile_w - 1)
            {

#if(NMS_WINDOW==3)
                //unrolled for 3x3
                int row_offset = (h_im + -1) * score_pitch;
                int col_offset = w_im + -1;
                int total_offset = row_offset + col_offset;

                score *= score >= score_data_gpu[total_offset];
                score *= score >= score_data_gpu[total_offset+1];
                score *= score >= score_data_gpu[total_offset+2];

                row_offset = (h_im + 0) * score_pitch;
                total_offset = row_offset + col_offset;

                score *= score >= score_data_gpu[total_offset];
                score *= score >= score_data_gpu[total_offset+1];
                score *= score >= score_data_gpu[total_offset+2];


                row_offset = (h_im + 1) * score_pitch;
                total_offset = row_offset + col_offset;

                score *= score >= score_data_gpu[total_offset];
                score *= score >= score_data_gpu[total_offset+1];
                score *= score >= score_data_gpu[total_offset+2];

#else

#ifdef DIVERGENCE_LESS_CODE

                //no divergence
                for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
                {
                    int row_offset = (h_im + i) * score_pitch;

                    for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                    {
                        int col_offset = w_im + j;

                        int res = (score >= score_data_gpu[row_offset + col_offset]);
                        score *= (res);
                    }
                }

#else
                // divergence
                for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
                {
                    int row_offset = (h_im + i) * score_pitch;

                    for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                    {
                        int col_offset = w_im + j;

                        if(score < score_data_gpu[row_offset + col_offset])
                        {
                            score = 0;
                            break;
                        }
                    }

                    if(!score)break;
                }
#endif

#endif
            }

            score_data_nms[h_im * score_nms_pitch + w_im] = score;

        }
    }


}

void ORB_GPU::FAST_apply_NMS_L(int height, int width,
                               int* score_data_gpu,
                               int score_pitch,
                               int* score_data_nms,
                               int score_nms_pitch,
                               cudaStream_t& cuda_stream)
{

    //    int roiheight = (height - 2*(BORDER_SKIP + NMS_WINDOW_HALF));
    //    int roiwidth  = (width - 2*(BORDER_SKIP + NMS_WINDOW_HALF));

    int roiheight = height;
    int roiwidth  = width;


#ifdef GRID_LAUNCH

    {
        int CUDA_NUM_BLOCKS_x = (roiwidth  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
        int CUDA_NUM_BLOCKS_y = (roiheight - 1) / CUDA_NUM_THREADS_PER_BLOCK_y + 1;

        dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
        dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, CUDA_NUM_THREADS_PER_BLOCK_y, 1);

        int n_threads =  roiheight * roiwidth;

        FASTapplyNMSGPU<<<grid_dim, block_dim, 0, cuda_stream>>>(
                                                                   n_threads,
                                                                   height, width,
                                                                   roiheight, roiwidth,
                                                                   score_data_gpu,
                                                                   score_pitch,
                                                                   score_data_nms,
                                                                   score_nms_pitch);


    }

#else
    {
        int n_threads =  roiheight * roiwidth;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1 ;

        FASTapplyNMSGPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                           n_threads,
                                                                                           height, width,
                                                                                           roiheight, roiwidth,
                                                                                           score_data_gpu,
                                                                                           score_data_nms);


    }

#endif
}

}

