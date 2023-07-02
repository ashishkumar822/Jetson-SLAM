#ifndef __TRACKING_GPU_H__
#define __TRACKING_GPU_H__


#include<vector>

namespace tracking_cuda{

#define STATIC_MEM_IS_IN

#ifdef STATIC_MEM_IS_IN

void compute_isInFrustum_GPU(int n_points,
                             float* Px_gpu, float* Py_gpu, float* Pz_gpu,
                             float* Pnx_gpu, float* Pny_gpu, float* Pnz_gpu,
                             float* MaxDistance_gpu,
                             float* invariance_maxDistance_gpu,
                             float* invariance_minDistance_gpu,
                             float* Rcw_gpu, float* tcw_gpu, float* Ow_gpu,
                             float& fx, float& fy, float& cx, float& cy,
                             int& minX, int& maxX, int& minY, int& maxY,
                             int& nScaleLevels,
                             float& logScaleFactor,
                             float& viewCosAngle,
                             float* invz_gpu, float* u_gpu, float* v_gpu,
                             int* predictedlevel_gpu,
                             float* viewCos_gpu,
                             unsigned char* is_infrustum_gpu);

#else

void compute_isInFrustum_GPU(std::vector<float>& Px, std::vector<float>& Py, std::vector<float>& Pz,
                             std::vector<float>& Pnx, std::vector<float>& Pny, std::vector<float>& Pnz,
                             std::vector<float>& MaxDistance,
                             std::vector<float>& invariance_maxDistance,
                             std::vector<float>& invariance_minDistance,
                             std::vector<float>& Rcw, std::vector<float>& tcw, std::vector<float>& Ow,
                             float& fx, float& fy, float& cx, float& cy,
                             int& minX, int& maxX, int& minY, int& maxY,
                             int& nScaleLevels,
                             float& logScaleFactor,
                             float& viewCosAngle,
                             std::vector<float>& invz, std::vector<float>& u, std::vector<float>& v,
                             std::vector<int>& predictedlevel,
                             std::vector<float>& viewCos,
                             std::vector<unsigned char>& is_infrustum);

}
#endif
}

#endif
