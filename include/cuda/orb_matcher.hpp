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



#ifndef __ORB_MATCHER_GPU_HPP__
#define __ORB_MATCHER_GPU_HPP__


#include<vector>

namespace orb_cuda{



void ORB_Search_by_projection_project_on_frame(int n_points,
                                             float* Px_gpu, float* Py_gpu, float* Pz_gpu,
                                             float* Rcw_gpu, float* tcw_gpu,
                                             float& fx, float& fy, float& cx, float& cy,
                                             float &minX, float &maxX, float &minY, float &maxY,
                                             float* u_gpu, float* v_gpu, float* invz_gpu,
                                             unsigned char* is_valid_gpu);

void ORB_compute_distances(int n_points,
                           int* idx_left, int* idx_right,
                           unsigned char*  descriptor_left,
                           unsigned char*  descriptor_right,
                           int* distance);


}

#endif
