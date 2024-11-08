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



void ORB_GPU::FAST_obtain_keypoints(void)
{
    if(!apply_nms_ms_)
    {
        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];
            cudaStreamSynchronize(cuda_stream);
        }
    }

    keypoints_.to_cpu(x_count_ + y_count_ + s_count_);

    // simply  copy the points to previous locations to remove non keypoints
    for(int i=0;i<n_levels_;i++)
    {
        int* kp_x = keypoints_.cpu_data() + x_offset_ + level_offset_[i];
        int* kp_y = keypoints_.cpu_data() + y_offset_ + level_offset_[i];
        int* kp_score = keypoints_.cpu_data() + s_offset_ + level_offset_[i];

        const int n_grids = n_tile_h_[i] * n_tile_w_[i];

        int count = 0;
        for(int j=0;j< n_grids;j++)
        {
            int score = kp_score[j];
            if(score > 0)
            {
                if( j != count)
                {
                    kp_x[count] = kp_x[j];
                    kp_y[count] = kp_y[j];
                    kp_score[count] = score;
                }


                count++;


            }
        }

        n_keypoints_[i] = count;
    }
}



}
