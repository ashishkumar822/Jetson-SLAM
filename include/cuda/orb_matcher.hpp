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
