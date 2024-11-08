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

//15.84us at 1235x375
//#define CUDA_NUM_THREADS_PER_BLOCK_x 32 // to ensure memory coalescing 16x16 can be an option however half of the warp will access different mem while anlther half the other
//#define CUDA_NUM_THREADS_PER_BLOCK_y 8

//14.04us at 1235x375
#define CUDA_NUM_THREADS_PER_BLOCK_x 32 // to ensure memory coalescing 16x16 can be an option however half of the warp will access different mem while anlther half the other
#define CUDA_NUM_THREADS_PER_BLOCK_y 8


#define CUDA_NUM_THREADS_PER_BLOCK 512


#define th_FAST_MIN 9 // 0.6
#define th_FAST_MAX 13 // 0.9


void ORB_GPU::patternCircle(int* pattern_circle, int rowStride, int patternSize)
{
    //    {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
    //    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}

    if(patternSize==16)
    {
        pattern_circle[0]  =  3*rowStride+0;
        pattern_circle[1]  =  3*rowStride+1;
        pattern_circle[2]  =  2*rowStride+2;
        pattern_circle[3]  =  1*rowStride+3;
        pattern_circle[4]  =  0*rowStride+3;
        pattern_circle[5]  = -1*rowStride+3;
        pattern_circle[6]  = -2*rowStride+2;
        pattern_circle[7]  = -3*rowStride+1;
        pattern_circle[8]  = -3*rowStride+0;
        pattern_circle[9]  = -3*rowStride-1;
        pattern_circle[10] = -2*rowStride-2;
        pattern_circle[11] = -1*rowStride-3;
        pattern_circle[12] =  0*rowStride-3;
        pattern_circle[13] =  1*rowStride-3;
        pattern_circle[14] =  2*rowStride-2;
        pattern_circle[15] =  3*rowStride-1;
    }
}



__global__ void FASTComputeScoreGPU_patternSize_16_lookup_mask(int imheight, int imwidth,
                                                               int threshold,
                                                               int* lookup_table,
                                                               const unsigned char* image_data,
                                                               int image_pitch,
                                                               const unsigned char* mask_data,
                                                               int mask_pitch,
                                                               int* score_data,
                                                               int score_pitch)
{

#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;


    if(w < imwidth && h < imheight)
    {
        const int h_im = h;// + BORDER_SKIP;
        const int w_im = w;// + BORDER_SKIP;

        const int offset = h_im * image_pitch + w_im;
        const int score_offset = h_im * score_pitch + w_im;

        if(!mask_data[h_im * mask_pitch + w_im])
            return;

#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int h = index / roiwidth;
        int w = index % roiwidth;
#endif

        if(w >= BORDER_SKIP && w < imwidth - BORDER_SKIP && h < imheight - BORDER_SKIP && h >= BORDER_SKIP)
        {

            int local_score = 0;

            const unsigned char* ptr = image_data + offset;

            const int v = ptr[0];

            const int vt = v + threshold;
            const int v_t = v - threshold;


            const int ptr_4  = ptr[3];
            const int ptr_12 = ptr[-3];


            if(ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t)
            {        score_data[score_offset] =  0;  return;}


            const int imw3 = 3*image_pitch;
            const int nimw3 = -imw3;

            const int ptr_0 = ptr[imw3];
            const int ptr_8 = ptr[nimw3];

            if(ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t)
            {        score_data[score_offset] =  0;return;}


            const int imw2 = 2*image_pitch;
            const int nimw2 = -imw2;
            const int nimw = -image_pitch;

            const int ptr_1  = ptr[imw3+1];
            const int ptr_2  = ptr[imw2+2];
            const int ptr_3  = ptr[image_pitch+3];
            const int ptr_5  = ptr[nimw+3];
            const int ptr_6  = ptr[nimw2+2];
            const int ptr_7  = ptr[nimw3+1];
            const int ptr_9  = ptr[nimw3-1];
            const int ptr_10 = ptr[nimw2-2];
            const int ptr_11 = ptr[nimw-3];
            const int ptr_13 = ptr[image_pitch-3];
            const int ptr_14 = ptr[imw2-2];
            const int ptr_15 = ptr[imw3-1];

            {
                {
                    int bright_idx = 0;
                    int dark_idx = 0;

                    {

                        if(ptr_0 > vt)  bright_idx  = 0x00000001;
                        if(ptr_1 > vt)  bright_idx |= 0x00000002;
                        if(ptr_2 > vt)  bright_idx |= 0x00000004;
                        if(ptr_3 > vt)  bright_idx |= 0x00000008;
                        if(ptr_4 > vt)  bright_idx |= 0x00000010;
                        if(ptr_5 > vt)  bright_idx |= 0x00000020;
                        if(ptr_6 > vt)  bright_idx |= 0x00000040;
                        if(ptr_7 > vt)  bright_idx |= 0x00000080;
                        if(ptr_8 > vt)  bright_idx |= 0x00000100;
                        if(ptr_9 > vt)  bright_idx |= 0x00000200;
                        if(ptr_10 > vt) bright_idx |= 0x00000400;
                        if(ptr_11 > vt) bright_idx |= 0x00000800;
                        if(ptr_12 > vt) bright_idx |= 0x00001000;
                        if(ptr_13 > vt) bright_idx |= 0x00002000;
                        if(ptr_14 > vt) bright_idx |= 0x00004000;
                        if(ptr_15 > vt) bright_idx |= 0x00008000;


                        if(ptr_0 < v_t)  dark_idx  = 0x00000001;
                        if(ptr_1 < v_t)  dark_idx |= 0x00000002;
                        if(ptr_2 < v_t)  dark_idx |= 0x00000004;
                        if(ptr_3 < v_t)  dark_idx |= 0x00000008;
                        if(ptr_4 < v_t)  dark_idx |= 0x00000010;
                        if(ptr_5 < v_t)  dark_idx |= 0x00000020;
                        if(ptr_6 < v_t)  dark_idx |= 0x00000040;
                        if(ptr_7 < v_t)  dark_idx |= 0x00000080;
                        if(ptr_8 < v_t)  dark_idx |= 0x00000100;
                        if(ptr_9 < v_t)  dark_idx |= 0x00000200;
                        if(ptr_10 < v_t) dark_idx |= 0x00000400;
                        if(ptr_11 < v_t) dark_idx |= 0x00000800;
                        if(ptr_12 < v_t) dark_idx |= 0x00001000;
                        if(ptr_13 < v_t) dark_idx |= 0x00002000;
                        if(ptr_14 < v_t) dark_idx |= 0x00004000;
                        if(ptr_15 < v_t) dark_idx |= 0x00008000;

                        if(lookup_table[bright_idx] || lookup_table[dark_idx])
                        {
                            local_score  = fabsf(ptr_0 - v) + fabsf(ptr_1 - v)
                                    + fabsf(ptr_2 - v) + fabsf(ptr_3 - v)
                                    + fabsf(ptr_4 - v) + fabsf(ptr_5 - v)
                                    + fabsf(ptr_6 - v) + fabsf(ptr_7 - v)
                                    + fabsf(ptr_8 - v) + fabsf(ptr_9 - v)
                                    + fabsf(ptr_10 - v) + fabsf(ptr_11 - v)
                                    + fabsf(ptr_12 - v) + fabsf(ptr_13 - v)
                                    + fabsf(ptr_14 - v) + fabsf(ptr_15 - v);
                        }

                    }
                }

                score_data[score_offset] = local_score;

            }
        }

    }
}



void ORB_GPU::FAST_compute_score_lookpup_mask(int height, int width,
                                              unsigned char* image_data_gpu,
                                              int image_pitch,
                                              unsigned char* mask_data_gpu,
                                              int mask_pitch,
                                              int threshold,
                                              int* lookup_table_gpu,
                                              int* score_data_gpu,
                                              int score_pitch,
                                              cudaStream_t& cuda_stream)
{
    {
        int CUDA_NUM_BLOCKS_x = (width  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
        int CUDA_NUM_BLOCKS_y = (height - 1) / CUDA_NUM_THREADS_PER_BLOCK_y + 1;

        dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
        dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, CUDA_NUM_THREADS_PER_BLOCK_y, 1);


        FASTComputeScoreGPU_patternSize_16_lookup_mask<<<grid_dim, block_dim,0 , cuda_stream>>>(
                                                                                                  height, width,
                                                                                                  threshold,
                                                                                                  lookup_table_gpu,
                                                                                                  image_data_gpu,
                                                                                                  image_pitch,
                                                                                                  mask_data_gpu,
                                                                                                  mask_pitch,
                                                                                                  score_data_gpu,
                                                                                                  score_pitch);

    }
}

}




