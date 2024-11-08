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


#define NMS_WINDOW 3
#define NMS_WINDOW_HALF (NMS_WINDOW - 1 ) / 2


void ORB_GPU::FAST_apply_NMS_MS_cpu(void)
{
    for(int i=0;i<n_levels_;i++)
    {
        int* kp_x = keypoints_.cpu_data() + x_offset_ + level_offset_[i];
        int* kp_y = keypoints_.cpu_data() + y_offset_ + level_offset_[i];
        int* kp_score = keypoints_.cpu_data() + s_offset_ + level_offset_[i];

        const int n_grids = n_tile_h_[i] * n_tile_w_[i];

        for(int j=0;j< n_grids;j++)
        {
            if(i==0)
            {
                nms_ms_cpu_x_[j].clear();
                nms_ms_cpu_y_[j].clear();
                nms_ms_cpu_score_[j].clear();
                nms_ms_cpu_level_[j].clear();
                nms_ms_cpu_idx_[j].clear();

                //// according to observationa at max 2 or 3 grids projects to the same level-0 grid
                //// max of 10 reserve to avoid push_back

                const int max_reserve = 10;

                nms_ms_cpu_x_[j].reserve(max_reserve);
                nms_ms_cpu_y_[j].reserve(max_reserve);
                nms_ms_cpu_score_[j].reserve(max_reserve);
                nms_ms_cpu_level_[j].reserve(max_reserve);
                nms_ms_cpu_idx_[j].reserve(max_reserve);
            }

            if(kp_score[j] > 0)
            {
                const int x_l0 = kp_x[j] * scale_[i] - BORDER_SKIP;
                const int y_l0 = kp_y[j] * scale_[i] - BORDER_SKIP;

                const int tile_x_l0 = x_l0 / tile_w_[0];
                const int tile_y_l0 = y_l0 / tile_h_[0];

                const int tile_idx = tile_y_l0 * n_tile_w_[0] + tile_x_l0;

                nms_ms_cpu_x_[tile_idx].push_back(x_l0);
                nms_ms_cpu_y_[tile_idx].push_back(y_l0);
                nms_ms_cpu_score_[tile_idx].push_back(kp_score[j]);
                nms_ms_cpu_level_[tile_idx].push_back(i);
                nms_ms_cpu_idx_[tile_idx].push_back(j);

                kp_score[j] = 0;
            }
        }
    }

    const int n_grids = n_tile_h_[0] * n_tile_w_[0];

    for(int i=0;i<n_grids;i++)
    {
        for(int j=0;j<nms_ms_cpu_score_[i].size();j++)
        {
            for(int k=0;k<nms_ms_cpu_score_[i].size();k++)
            {
                int my_level = nms_ms_cpu_level_[i][j];
                int other_level = nms_ms_cpu_level_[i][k];

                if( j==k || my_level == other_level)
                    continue;

                if(nms_ms_cpu_score_[i][j] && nms_ms_cpu_score_[i][k])
                {
                    int x_j = nms_ms_cpu_x_[i][j];
                    int y_j = nms_ms_cpu_y_[i][j];

                    int x_k = nms_ms_cpu_x_[i][k];
                    int y_k = nms_ms_cpu_y_[i][k];

                    int x_diff = (x_j - x_k);
                    int y_diff = (y_j - y_k);

                    if(x_diff >= -NMS_WINDOW_HALF && x_diff <= NMS_WINDOW_HALF && y_diff >= -NMS_WINDOW_HALF && y_diff <= NMS_WINDOW_HALF)
                    {
                        if(nms_ms_cpu_score_[i][j] < nms_ms_cpu_score_[i][k])
                        {
                            nms_ms_cpu_score_[i][j] = 0;
                        }
                        else
                            nms_ms_cpu_score_[i][k] = 0;
                    }
                }
            }
        }
    }

    for(int i=0;i<n_grids;i++)
    {
        for(int j=0;j<nms_ms_cpu_score_[i].size();j++)
        {
            int level = nms_ms_cpu_level_[i][j];
            int idx = nms_ms_cpu_idx_[i][j];

            int* kp_score = keypoints_.cpu_data() + s_offset_ + level_offset_[level];

            if(nms_ms_cpu_score_[i][j] != 0)
            {
                kp_score[idx] = nms_ms_cpu_score_[i][j];
            }
        }
    }
}



}
