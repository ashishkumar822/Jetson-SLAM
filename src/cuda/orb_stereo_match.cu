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

//// Part of the Middle-end

#define STATIC_MEM


#ifdef STATIC_MEM

#include "cuda/orb_gpu.hpp"

#include<vector>
#include<opencv2/opencv.hpp>

#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<cublas_v2.h>

#include<pcl/console/time.h>

#include<chrono>

namespace orb_cuda {

#define CUDA_NUM_THREADS_PER_BLOCK 512




__global__ void ORBGetDistanceStereoGPU(int n_threads,
                                        int* idx_left, int* idx_right,
                                        unsigned char*  descriptor_left,
                                        unsigned char*  descriptor_right,
                                        int* distance)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int* d_left  = (int*)(descriptor_left  + idx_left[index] * 32);
        int* d_right = (int*)(descriptor_right + idx_right[index] * 32);

        int dist=0;

        for(int i=0; i<8; i++)
        {
            unsigned  int v = d_left[i] ^ d_right[i];
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        distance[index] = dist;
    }
}


#define PATCH_WINDOW 11
#define PATCH_WINDOW_HALF ((PATCH_WINDOW - 1) / 2)
#define PATCH_WINDOW_SIZE (PATCH_WINDOW * PATCH_WINDOW)

#define NBRHOOD 11
#define NBRHOOD_HALF ((NBRHOOD - 1) / 2)


__global__ void Compute_L1_distance_GPU(int n_threads,
                                        int* height, int* width,
                                        int* left_keypoint_x,
                                        int* right_keypoint_x,
                                        int* keypoint_y,
                                        unsigned char** image_left,
                                        unsigned char** image_right,
                                        int* octave,
                                        float* L1_distance_vector)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int idx = index / (PATCH_WINDOW_SIZE * NBRHOOD);
        int left_x = left_keypoint_x[idx];
        int right_x = right_keypoint_x[idx];
        int y = keypoint_y[idx];

        int im_octave = octave[idx];
        //        int im_height = height[im_octave];
        int im_width  = width[im_octave];

        int nbr_idx = ((index / PATCH_WINDOW_SIZE) % NBRHOOD) - NBRHOOD_HALF;

        unsigned char* left_im = image_left[im_octave] + y * im_width + left_x;
        unsigned char* right_im = image_right[im_octave] + y * im_width + right_x + nbr_idx;

        float left_center_val = left_im[0];
        float right_center_val = right_im[0];

        int win_h = (index / PATCH_WINDOW) % PATCH_WINDOW;
        int win_w = (index % PATCH_WINDOW);

        const int offset = (win_h - PATCH_WINDOW_HALF) * im_width + win_w - PATCH_WINDOW_HALF;

        L1_distance_vector[index] = fabsf((left_im[offset] - left_center_val)-(right_im[offset]-right_center_val));
    }
}


void ORB_GPU::ORB_compute_stereo_match(int ORB_TH_HIGH, int ORB_TH_LOW,
                                       float mb, float mbf,
                                       std::vector<int>& octave_height,
                                       std::vector<int>& octave_width,
                                       std::vector<cv::KeyPoint>& mvKeys,
                                       std::vector<cv::KeyPoint>& mvKeysRight,
                                       std::vector<float>& mvuRight,
                                       std::vector<float>& mvDepth,
                                       unsigned char* descriptor_left_gpu,
                                       unsigned char* descriptor_right_gpu,
                                       std::vector<SyncedMem<unsigned char> >& images_left_smem,
                                       std::vector<SyncedMem<unsigned char> >& images_right_smem)
{

    const int nRows = octave_height[0];

    //Assign keypoints to row table
    std::vector<std::vector<int> > vRowIndices(nRows,std::vector<int>());

    const int Nr = mvKeysRight.size();

    int mcount= 0;
    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f * scale_[kp.octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi = minr;yi <= maxr; yi++)
        {
            vRowIndices[yi].push_back(iR);
            mcount++;
        }
    }

//    std::cout << "mtccg = " <<mcount << "\n";

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;


    std::vector<int> left_keypoints_idx;
    std::vector<int> right_keypoints_idx;

    for(int i=0;i<mvKeys.size();i++)
    {
        const cv::KeyPoint &kpL = mvKeys[i];
        const int levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        std::vector<int>& indices = vRowIndices[vL];

        for(int j=0;j<indices.size();j++)
        {
            const size_t right_idx = indices[j];
            const cv::KeyPoint &kpR = mvKeysRight[right_idx];

            if(kpR.octave < levelL-1 || kpR.octave > levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR >= minU && uR <= maxU)
            {
                left_keypoints_idx.push_back(i);
                right_keypoints_idx.push_back(right_idx);
            }
        }
    }


    int n_points_to_match = left_keypoints_idx.size();

//    std::cout << "Match points = " << n_points_to_match << "\n";

    static SyncedMem<int> idx_left;
    static SyncedMem<int> idx_right;
    static SyncedMem<int> distances;

    idx_left.resize(left_keypoints_idx.size());
    idx_right.resize(left_keypoints_idx.size());
    distances.resize(left_keypoints_idx.size());

    // compute the distances between left-right pair on GPU
    {
        int* idx_left_gpu = idx_left.gpu_data();
        int* idx_right_gpu = idx_right.gpu_data();
        int* distances_gpu = distances.gpu_data();

        cudaMemcpy(idx_left_gpu, left_keypoints_idx.data(), sizeof(int)*left_keypoints_idx.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(idx_right_gpu, right_keypoints_idx.data(), sizeof(int)*left_keypoints_idx.size(),cudaMemcpyHostToDevice);

        int n_threads =  n_points_to_match;

        int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

        ORBGetDistanceStereoGPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK>>>(
                                                                                   n_threads,
                                                                                   idx_left_gpu, idx_right_gpu,
                                                                                   descriptor_left_gpu,
                                                                                   descriptor_right_gpu,
                                                                                   distances_gpu
                                                                                   );


        cudaStreamSynchronize(0);
        distances.to_cpu_async();
    }

//    std::cout << "Computed descriptor distance\n";

    int thOrbDist = (ORB_TH_HIGH + ORB_TH_LOW ) / 2;

    // For each left keypoint search a match in the right image
    // Find all keypoints with distance less than thOrbDist
    static SyncedMem<int> corr_match_left_idx;
    static SyncedMem<int> corr_match_right_idx;
    static SyncedMem<int> corr_match_octave;
    static SyncedMem<int> corr_match_x_left;
    static SyncedMem<int> corr_match_x_right;
    static SyncedMem<int> corr_match_y;

    const int N_kp = mvKeys.size();

    corr_match_left_idx.resize(N_kp);
    corr_match_right_idx.resize(N_kp);
    corr_match_octave.resize(N_kp);
    corr_match_x_left.resize(N_kp);
    corr_match_x_right.resize(N_kp);
    corr_match_y.resize(N_kp);

    int n_corr_match = 0;

    {
        distances.sync_stream();
        int* distances_cpu = distances.cpu_data();

        std::vector<int> match_right_idx(mvKeys.size(),-1);
        std::vector<int> match_distances(mvKeys.size(), ORB_TH_HIGH);

        for(int i=0;i<left_keypoints_idx.size();i++)
        {
            int& left_idx = left_keypoints_idx[i];
            int& right_idx = right_keypoints_idx[i];

            int& match_distance = match_distances[left_idx];
            int& distance = distances_cpu[i];


            if(distance < match_distance)
            {
                match_distance = distance;
                match_right_idx[left_idx] = right_idx;
            }
        }

        int* corr_match_left_idx_data = corr_match_left_idx.cpu_data();
        int* corr_match_right_idx_data = corr_match_right_idx.cpu_data();
        int* corr_match_octave_data = corr_match_octave.cpu_data();
        int* corr_match_x_left_data = corr_match_x_left.cpu_data();
        int* corr_match_x_right_data = corr_match_x_right.cpu_data();
        int* corr_match_y_data = corr_match_y.cpu_data();

        int L = 5;
        int w = 5;

        for(int i=0;i<match_right_idx.size();i++)
        {
            if(match_right_idx[i] != -1)
            {
                if(match_distances[i] < thOrbDist)
                {
                    int bestIdxR = match_right_idx[i];

                    const cv::KeyPoint &kpL = mvKeys[i];
                    const cv::KeyPoint &kpR = mvKeysRight[bestIdxR];


                    const float vL0 = kpL.pt.y;
                    const float uL0 = kpL.pt.x;
                    const float uR0 = kpR.pt.x;
                    const float scaleFactor = inv_scale_[kpL.octave];
                    const float scaleduR0 = round(uR0 * scaleFactor);
                    const float scaleduL0 = round(uL0 * scaleFactor);
                    const float scaledvL0 = round(vL0 * scaleFactor);

                    const float iniu = scaleduR0 - L - w;
                    const float endu = scaleduR0 + L + w;

                    if(iniu < 0 || endu >= width_[kpL.octave])
                        continue;

                    corr_match_left_idx_data[n_corr_match] = i;
                    corr_match_right_idx_data[n_corr_match] = match_right_idx[i];
                    corr_match_octave_data[n_corr_match] = kpL.octave;
                    corr_match_x_left_data[n_corr_match] = scaleduL0;
                    corr_match_x_right_data[n_corr_match] = scaleduR0;
                    corr_match_y_data[n_corr_match] = scaledvL0;

                    n_corr_match++;
                }
            }
        }
    }

    // resizing won't allocate new memory.
    // It will just change count_ because n_corr_match <= mvKeys.size()
    corr_match_left_idx.resize(n_corr_match);
    corr_match_right_idx.resize(n_corr_match);
    corr_match_octave.resize(n_corr_match);
    corr_match_x_left.resize(n_corr_match);
    corr_match_x_right.resize(n_corr_match);
    corr_match_y.resize(n_corr_match);

//    std::cout << "correlation match points = " << n_corr_match << "\n";

//    std::cout << "Filled points for Window search\n";

    static SyncedMem<float> distance_L1;


    // compute window search for corr_match_left_idx in the neighborhood of
    // corr_match_right_idx
    // depth will be computed only if the window L1-distance is smaller than a threshold
    {
        corr_match_octave.to_gpu_async();
        corr_match_x_left.to_gpu_async();
        corr_match_x_right.to_gpu_async();
        corr_match_y.to_gpu_async();


        std::vector<unsigned char*> images_left_gpu_data;
        std::vector<unsigned char*> images_right_gpu_data;

        for(int i=0;i<images_left_smem.size();i++)
        {
            images_left_gpu_data.push_back(images_left_smem[i].gpu_data());
            images_right_gpu_data.push_back(images_right_smem[i].gpu_data());
        }

        unsigned char** images_left = images_left_gpu_data.data();
        unsigned char** images_right = images_right_gpu_data.data();

        unsigned char** images_left_gpu;
        unsigned char** images_right_gpu;

        int* height_gpu;
        int* width_gpu;

        cudaMalloc((void**)&height_gpu, sizeof(int) * octave_height.size());
        cudaMalloc((void**)&width_gpu, sizeof(int) * octave_width.size());

        cudaMalloc((void**)&images_left_gpu, sizeof(unsigned char*) * octave_height.size());
        cudaMalloc((void**)&images_right_gpu, sizeof(unsigned char*) * octave_width.size());

        cudaMemcpy(height_gpu, octave_height.data(), sizeof(int)*octave_height.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(width_gpu, octave_width.data(), sizeof(int)*octave_width.size(),cudaMemcpyHostToDevice);

        cudaMemcpy(images_left_gpu, images_left, sizeof(unsigned char*)*octave_height.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(images_right_gpu, images_right, sizeof(unsigned char*)*octave_width.size(),cudaMemcpyHostToDevice);

        corr_match_octave.sync_stream();
        corr_match_x_left.sync_stream();
        corr_match_x_right.sync_stream();
        corr_match_y.sync_stream();


        static SyncedMem<float> ones;
        static bool ones_once = false;
        if(!ones_once)
        {
            ones.resize(PATCH_WINDOW_SIZE);
            for(int i=0;i<PATCH_WINDOW_SIZE;i++)
                ones.cpu_data()[i] = 1.0;
            ones.to_gpu();
            ones_once = true;
        }

        int n_count_distance_vector = n_corr_match * PATCH_WINDOW_SIZE * NBRHOOD;
        int n_count_L1 = n_corr_match * NBRHOOD;

        static SyncedMem<float> distance_L1_vector;
        distance_L1_vector.resize(n_count_distance_vector);
        distance_L1.resize(n_count_L1);

        // L1 distance vector
        {
            int n_threads = n_corr_match * PATCH_WINDOW_SIZE * NBRHOOD;

            int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

            Compute_L1_distance_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK>>>(
                                                                                       n_threads,
                                                                                       height_gpu, width_gpu,
                                                                                       corr_match_x_left.gpu_data(),
                                                                                       corr_match_x_right.gpu_data(),
                                                                                       corr_match_y.gpu_data(),
                                                                                       images_left_gpu,
                                                                                       images_right_gpu,
                                                                                       corr_match_octave.gpu_data(),
                                                                                       distance_L1_vector.gpu_data()
                                                                                       );

//            std::cout << "cuda last error = " << cudaGetLastError() << "\n";

            cudaStreamSynchronize(0);
        }
//        std::cout << "cuda last error = " << cudaGetLastError() << "\n";

//        std::cout << "Computed distance vector\n";

        {
            cublasHandle_t cublas_handle;
            cublasCreate_v2(&cublas_handle);

            // reverse order of a and B because of column major. since we have row major
            // M*k * k*N
            //            int M = 1;
            //            int N = corr_match_left_idx.size() * NBRHOOD;
            //            int k = PATCH_WINDOW_SIZE;

            //            int lda = PATCH_WINDOW_SIZE;
            //            int ldb = PATCH_WINDOW_SIZE;
            //            int ldc = 1;

            //            float alpha = 1.0f;
            //            float beta = 0.0f;

            //            cublasSgemm_v2(cublas_handle,
            //                           CUBLAS_OP_N,CUBLAS_OP_N,
            //                           M, N, k,
            //                           &alpha,
            //                           ones_gpu, lda,
            //                           distance_L1_vector, ldb,
            //                           &beta,
            //                           distance_L1_gpu,ldc);

            int M = n_corr_match * NBRHOOD;
            int N = PATCH_WINDOW_SIZE;

            int lda = PATCH_WINDOW_SIZE;
            int ldb = 1;
            int ldc = 1;

            float alpha = 1.0f;
            float beta = 0.0f;

            cublasSgemv_v2(cublas_handle,
                           CUBLAS_OP_T,
                           N, M,
                           &alpha,
                           distance_L1_vector.gpu_data(), lda,
                           ones.gpu_data(), ldb,
                           &beta,
                           distance_L1.gpu_data(),ldc);

//            std::cout << "Computed Multiply\n";

//            std::cout << "cuda last error = " << cudaGetLastError() << "\n";

            distance_L1.to_cpu();

            cublasDestroy_v2(cublas_handle);
        }

//        std::cout << "Computed L1 distances\n";

        cudaFree(height_gpu);
        cudaFree(width_gpu);

        cudaFree(images_left_gpu);
        cudaFree(images_right_gpu);
    }


    {
        std::vector<std::pair<int, int> > vDistIdx;
        vDistIdx.reserve(mvKeys.size());


        mvuRight.resize(mvKeys.size(), -1.0f);
        mvDepth.resize(mvKeys.size(), -1.0f);

        float* distance_L1_cpu = distance_L1.cpu_data();
        int* corr_match_left_idx_cpu = corr_match_left_idx.cpu_data();
        int* corr_match_right_idx_cpu = corr_match_right_idx.cpu_data();

        for(int i=0;i<n_corr_match;i++)
        {
            int bestDist = INT_MAX;
            int bestR = 0;

            int offset = i * NBRHOOD;

            for(int l=0; l< NBRHOOD; l++)
            {
                float dist = distance_L1_cpu[offset + l];
                if(dist < bestDist)
                {
                    bestDist = dist;
                    bestR = l;
                }
            }

            if(bestR== 0 || bestR== NBRHOOD-1)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = distance_L1_cpu[offset + bestR-1];
            const float dist2 = distance_L1_cpu[offset + bestR];
            const float dist3 = distance_L1_cpu[offset + bestR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;


            int left_idx = corr_match_left_idx_cpu[i];
            int right_idx = corr_match_right_idx_cpu[i];

            const cv::KeyPoint &kpL = mvKeys[left_idx];
            const cv::KeyPoint &kpR = mvKeysRight[right_idx];

            const float uL = kpL.pt.x;
            const float uR0 = kpR.pt.x;

            const float scaleFactor = inv_scale_[kpL.octave];
            const float scaleduR0 = round(uR0 * scaleFactor);

            // Re-scaled coordinate
            float bestuR = scale_[kpL.octave]*((float)scaleduR0+(float)bestR-NBRHOOD_HALF+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[left_idx]=mbf/disparity;
                mvuRight[left_idx] = bestuR;
                vDistIdx.push_back(std::pair<int,int>(bestDist, left_idx));
            }
        }


        sort(vDistIdx.begin(),vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size()/2].first;
        const float thDist = 1.5f*1.4f*median;

        for(int i=vDistIdx.size()-1;i>=0;i--)
        {
            if(vDistIdx[i].first<thDist)
                break;
            else
            {
                mvuRight[vDistIdx[i].second]=-1;
                mvDepth[vDistIdx[i].second]=-1;
            }
        }
    }
}


}


#else

#include "cuda/orb_gpu.hpp"

#include<vector>
#include<opencv2/opencv.hpp>

#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<cublas_v2.h>

#include<pcl/console/time.h>

#include<chrono>

namespace orb_cuda {

#define CUDA_NUM_THREADS_PER_BLOCK 512




__global__ void ORBGetDistanceStereoGPU(int n_threads,
                                        int* idx_left, int* idx_right,
                                        unsigned char*  descriptor_left,
                                        unsigned char*  descriptor_right,
                                        int* distance)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int* d_left  = (int*)(descriptor_left  + idx_left[index] * 32);
        int* d_right = (int*)(descriptor_right + idx_right[index] * 32);

        int dist=0;

        for(int i=0; i<8; i++)
        {
            unsigned  int v = d_left[i] ^ d_right[i];
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        distance[index] = dist;
    }
}


#define PATCH_WINDOW 11
#define PATCH_WINDOW_HALF ((PATCH_WINDOW - 1) / 2)
#define PATCH_WINDOW_SIZE (PATCH_WINDOW * PATCH_WINDOW)

#define NBRHOOD 11
#define NBRHOOD_HALF ((NBRHOOD - 1) / 2)


__global__ void Compute_L1_distance_GPU(int n_threads,
                                        int* height, int* width,
                                        int* left_keypoint_x,
                                        int* right_keypoint_x,
                                        int* keypoint_y,
                                        unsigned char** image_left,
                                        unsigned char** image_right,
                                        int* octave,
                                        float* L1_distance_vector)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int idx = index / (PATCH_WINDOW_SIZE * NBRHOOD);
        int left_x = left_keypoint_x[idx];
        int right_x = right_keypoint_x[idx];
        int y = keypoint_y[idx];

        int im_octave = octave[idx];
        //        int im_height = height[im_octave];
        int im_width  = width[im_octave];

        int nbr_idx = ((index / PATCH_WINDOW_SIZE) % NBRHOOD) - NBRHOOD_HALF;

        unsigned char* left_im = image_left[im_octave] + y * im_width + left_x;
        unsigned char* right_im = image_right[im_octave] + y * im_width + right_x + nbr_idx;

        float left_center_val = left_im[0];
        float right_center_val = right_im[0];

        int win_h = (index / PATCH_WINDOW) % PATCH_WINDOW;
        int win_w = (index % PATCH_WINDOW);

        const int offset = (win_h - PATCH_WINDOW_HALF) * im_width + win_w - PATCH_WINDOW_HALF;

        L1_distance_vector[index] = fabsf((left_im[offset] - left_center_val)-(right_im[offset]-right_center_val));
    }
}


void ORB_GPU::ORB_compute_stereo_match(int ORB_TH_HIGH, int ORB_TH_LOW,
                                       float mb, float mbf,
                                       std::vector<int>& octave_height,
                                       std::vector<int>& octave_width,
                                       std::vector<cv::KeyPoint>& mvKeys,
                                       std::vector<cv::KeyPoint>& mvKeysRight,
                                       std::vector<float>& mvuRight,
                                       std::vector<float>& mvDepth,
                                       unsigned char* keypoint_descriptor_left,
                                       unsigned char* keypoint_descriptor_right,
                                       std::vector<SyncedMem<unsigned char>* >& images_left_smem,
                                       std::vector<SyncedMem<unsigned char>* >& images_right_smem)
{

    const int nRows = octave_height[0];

    //Assign keypoints to row table
    std::vector<std::vector<int> > vRowIndices(nRows,std::vector<int>());

    const int Nr = mvKeysRight.size();

    int mcount= 0;
    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f * scale_[kp.octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi = minr;yi <= maxr; yi++)
        {
            vRowIndices[yi].push_back(iR);
            mcount++;
        }
    }

//    std::cout << "mtccg = " <<mcount << "\n";

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;


    std::vector<int> left_keypoints_idx;
    std::vector<int> right_keypoints_idx;

    for(int i=0;i<mvKeys.size();i++)
    {
        const cv::KeyPoint &kpL = mvKeys[i];
        const int levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        std::vector<int>& indices = vRowIndices[vL];

        for(int j=0;j<indices.size();j++)
        {
            const size_t right_idx = indices[j];
            const cv::KeyPoint &kpR = mvKeysRight[right_idx];

            if(kpR.octave < levelL-1 || kpR.octave > levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR >= minU && uR <= maxU)
            {
                left_keypoints_idx.push_back(i);
                right_keypoints_idx.push_back(right_idx);
            }
        }
    }


    int n_points_to_match = left_keypoints_idx.size();

//    std::cout << "Match points = " << n_points_to_match << "\n";

    std::vector<int> distances(left_keypoints_idx.size());

    // compute the distances between left-right pair on GPU
    {
        int* idx_left_gpu;
        int* idx_right_gpu;
        int* distances_gpu;

        unsigned char* descriptor_left_gpu;
        unsigned char* descriptor_right_gpu;

        cudaMalloc((void**)&idx_left_gpu, sizeof(int) * left_keypoints_idx.size());
        cudaMalloc((void**)&idx_right_gpu, sizeof(int) * left_keypoints_idx.size());
        cudaMalloc((void**)&distances_gpu, sizeof(int) * left_keypoints_idx.size());

        cudaMalloc((void**)&descriptor_left_gpu, sizeof(unsigned char) * 32*mvKeys.size());
        cudaMalloc((void**)&descriptor_right_gpu, sizeof(unsigned char) *32* mvKeysRight.size());


        cudaMemcpy(idx_left_gpu, left_keypoints_idx.data(), sizeof(int)*left_keypoints_idx.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(idx_right_gpu, right_keypoints_idx.data(), sizeof(int)*left_keypoints_idx.size(),cudaMemcpyHostToDevice);

        cudaMemcpy(descriptor_left_gpu, keypoint_descriptor_left, sizeof(unsigned char)*32*mvKeys.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(descriptor_right_gpu, keypoint_descriptor_right, sizeof(unsigned char)*32*mvKeysRight.size(),cudaMemcpyHostToDevice);

        int n_threads =  n_points_to_match;

        int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

        ORBGetDistanceStereoGPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK>>>(
                                                                                   n_threads,
                                                                                   idx_left_gpu, idx_right_gpu,
                                                                                   descriptor_left_gpu,
                                                                                   descriptor_right_gpu,
                                                                                   distances_gpu
                                                                                   );


        cudaStreamSynchronize(0);

        cudaMemcpy(distances.data(), distances_gpu, sizeof(int)*left_keypoints_idx.size(),cudaMemcpyDeviceToHost);

        cudaFree(idx_left_gpu);
        cudaFree(idx_right_gpu);
        cudaFree(distances_gpu);

        cudaFree(descriptor_left_gpu);
        cudaFree(descriptor_right_gpu);
    }

//    std::cout << "Computed descriptor distance\n";

    int thOrbDist = (ORB_TH_HIGH + ORB_TH_LOW ) / 2;

    // For each left keypoint search a match in the right image
    // Find all keypoints with distance less than thOrbDist
    std::vector<int> corr_match_left_idx;
    std::vector<int> corr_match_right_idx;
    std::vector<int> corr_match_octave;
    std::vector<int> corr_match_x_left;
    std::vector<int> corr_match_x_right;
    std::vector<int> corr_match_y;

    {
        std::vector<int> match_right_idx(mvKeys.size(),-1);
        std::vector<int> match_distances(mvKeys.size(), ORB_TH_HIGH);

        for(int i=0;i<left_keypoints_idx.size();i++)
        {
            int& left_idx = left_keypoints_idx[i];
            int& right_idx = right_keypoints_idx[i];

            int& match_distance = match_distances[left_idx];
            int& distance = distances[i];


            if(distance < match_distance)
            {
                match_distance = distance;
                match_right_idx[left_idx] = right_idx;
            }
        }


        int L = 5;
        int w = 5;

        for(int i=0;i<match_right_idx.size();i++)
        {
            if(match_right_idx[i] != -1)
            {
                if(match_distances[i] < thOrbDist)
                {
                    int bestIdxR = match_right_idx[i];

                    const cv::KeyPoint &kpL = mvKeys[i];
                    const cv::KeyPoint &kpR = mvKeysRight[bestIdxR];


                    const float vL0 = kpL.pt.y;
                    const float uL0 = kpL.pt.x;
                    const float uR0 = kpR.pt.x;
                    const float scaleFactor = inv_scale_[kpL.octave];
                    const float scaleduR0 = round(uR0 * scaleFactor);
                    const float scaleduL0 = round(uL0 * scaleFactor);
                    const float scaledvL0 = round(vL0 * scaleFactor);

                    const float iniu = scaleduR0 - L - w;
                    const float endu = scaleduR0 + L + w;

                    if(iniu < 0 || endu >= width_[kpL.octave])
                        continue;

                    corr_match_left_idx.push_back(i);
                    corr_match_right_idx.push_back(match_right_idx[i]);
                    corr_match_octave.push_back(kpL.octave);
                    corr_match_x_left.push_back(scaleduL0);
                    corr_match_x_right.push_back(scaleduR0);
                    corr_match_y.push_back(scaledvL0);
                }
            }
        }
    }

//    std::cout << "correlation match points = " << corr_match_left_idx.size() << "\n";

//    std::cout << "Filled points for Window search\n";

    std::vector<float> distance_L1(corr_match_left_idx.size() * NBRHOOD);

    // compute window search for corr_match_left_idx in the neighborhood of
    // corr_match_right_idx
    // depth will be computed only if the window L1-distance is smaller than a threshold
    {
        std::vector<unsigned char*> images_left_gpu_data;
        std::vector<unsigned char*> images_right_gpu_data;

        for(int i=0;i<images_left_smem.size();i++)
        {
            images_left_gpu_data.push_back(images_left_smem[i]->gpu_data());
            images_right_gpu_data.push_back(images_right_smem[i]->gpu_data());
        }

        unsigned char** images_left = images_left_gpu_data.data();
        unsigned char** images_right = images_right_gpu_data.data();


        int* corr_octave_gpu;
        int* corr_x_left_gpu;
        int* corr_x_right_gpu;
        int* corr_y_gpu;
        int* corr_distances_gpu;
        int* height_gpu;
        int* width_gpu;

        unsigned char** images_left_gpu;
        unsigned char** images_right_gpu;

        //        std::vector<unsigned char*> im_left_gpu(octave_height.size());
        //        std::vector<unsigned char*> im_right_gpu(octave_height.size());


        cudaMalloc((void**)&corr_octave_gpu, sizeof(int) * corr_match_left_idx.size());
        cudaMalloc((void**)&corr_x_left_gpu, sizeof(int) * corr_match_left_idx.size());
        cudaMalloc((void**)&corr_x_right_gpu, sizeof(int) * corr_match_left_idx.size());
        cudaMalloc((void**)&corr_y_gpu, sizeof(int) * corr_match_left_idx.size());
        cudaMalloc((void**)&corr_distances_gpu, sizeof(int) * corr_match_left_idx.size());
        cudaMalloc((void**)&height_gpu, sizeof(int) * octave_height.size());
        cudaMalloc((void**)&width_gpu, sizeof(int) * octave_width.size());

        cudaMalloc((void**)&images_left_gpu, sizeof(unsigned char*) * octave_height.size());
        cudaMalloc((void**)&images_right_gpu, sizeof(unsigned char*) * octave_width.size());

        //        for(int i=0;i<im_left_gpu.size();i++)
        //        {
        //            cudaMalloc((void**)&im_left_gpu[i], sizeof(unsigned char) * octave_height[i] * octave_width[i]);
        //            cudaMalloc((void**)&im_right_gpu[i], sizeof(unsigned char) * octave_height[i] * octave_width[i]);

        //            cudaMemcpy(im_left_gpu[i], images_left[i], sizeof(unsigned char)* octave_height[i] * octave_width[i],cudaMemcpyHostToDevice);
        //            cudaMemcpy(im_right_gpu[i], images_right[i], sizeof(unsigned char)* octave_height[i] * octave_width[i],cudaMemcpyHostToDevice);
        //        }

        cudaMemcpy(corr_octave_gpu, corr_match_octave.data(), sizeof(int)*corr_match_left_idx.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(corr_x_left_gpu, corr_match_x_left.data(), sizeof(int)*corr_match_left_idx.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(corr_x_right_gpu, corr_match_x_right.data(), sizeof(int)*corr_match_left_idx.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(corr_y_gpu, corr_match_y.data(), sizeof(int)*corr_match_left_idx.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(height_gpu, octave_height.data(), sizeof(int)*octave_height.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(width_gpu, octave_width.data(), sizeof(int)*octave_width.size(),cudaMemcpyHostToDevice);

        cudaMemcpy(images_left_gpu, images_left, sizeof(unsigned char*)*octave_height.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(images_right_gpu, images_right, sizeof(unsigned char*)*octave_width.size(),cudaMemcpyHostToDevice);

        //        cudaMemcpy(images_left_gpu, im_left_gpu.data(), sizeof(unsigned char*)*octave_height.size(),cudaMemcpyHostToDevice);
        //        cudaMemcpy(images_right_gpu, im_right_gpu.data(), sizeof(unsigned char*)*octave_width.size(),cudaMemcpyHostToDevice);


        float* distance_L1_vector;
        float* distance_L1_gpu;
        float* ones_gpu;

        std::vector<float> ones(PATCH_WINDOW_SIZE, 1.0f);

        int n_count_distance_vector = corr_match_left_idx.size() * PATCH_WINDOW_SIZE * NBRHOOD;
        int n_count_L1 = corr_match_left_idx.size() * NBRHOOD;

        cudaMalloc((void**)&distance_L1_vector, sizeof(float) * n_count_distance_vector);
        cudaMalloc((void**)&distance_L1_gpu, sizeof(float) * n_count_L1);

        cudaMalloc((void**)&ones_gpu, sizeof(float) * ones.size());
        cudaMemcpy(ones_gpu, ones.data(), sizeof(float)*ones.size(),cudaMemcpyHostToDevice);

        // L1 distance vector
        {
            int n_threads = corr_match_left_idx.size() * PATCH_WINDOW_SIZE * NBRHOOD;

            int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

            Compute_L1_distance_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK>>>(
                                                                                       n_threads,
                                                                                       height_gpu, width_gpu,
                                                                                       corr_x_left_gpu,
                                                                                       corr_x_right_gpu,
                                                                                       corr_y_gpu,
                                                                                       images_left_gpu,
                                                                                       images_right_gpu,
                                                                                       corr_octave_gpu,
                                                                                       distance_L1_vector
                                                                                       );

//            std::cout << "cuda last error = " << cudaGetLastError() << "\n";

            cudaStreamSynchronize(0);
        }
//        std::cout << "cuda last error = " << cudaGetLastError() << "\n";

//        std::cout << "Computed distance vector\n";

        {
            cublasHandle_t cublas_handle;
            cublasCreate_v2(&cublas_handle);

            // reverse order of a and B because of column major. since we have row major
            // M*k * k*N
            //            int M = 1;
            //            int N = corr_match_left_idx.size() * NBRHOOD;
            //            int k = PATCH_WINDOW_SIZE;

            //            int lda = PATCH_WINDOW_SIZE;
            //            int ldb = PATCH_WINDOW_SIZE;
            //            int ldc = 1;

            //            float alpha = 1.0f;
            //            float beta = 0.0f;

            //            cublasSgemm_v2(cublas_handle,
            //                           CUBLAS_OP_N,CUBLAS_OP_N,
            //                           M, N, k,
            //                           &alpha,
            //                           ones_gpu, lda,
            //                           distance_L1_vector, ldb,
            //                           &beta,
            //                           distance_L1_gpu,ldc);

            int M = corr_match_left_idx.size() * NBRHOOD;
            int N = PATCH_WINDOW_SIZE;

            int lda = PATCH_WINDOW_SIZE;
            int ldb = 1;
            int ldc = 1;

            float alpha = 1.0f;
            float beta = 0.0f;

            cublasSgemv_v2(cublas_handle,
                           CUBLAS_OP_T,
                           N, M,
                           &alpha,
                           distance_L1_vector, lda,
                           ones_gpu, ldb,
                           &beta,
                           distance_L1_gpu,ldc);

//            std::cout << "Computed Multiply\n";

//            std::cout << "cuda last error = " << cudaGetLastError() << "\n";

            cudaMemcpy(distance_L1.data(), distance_L1_gpu, sizeof(float)*corr_match_left_idx.size() * NBRHOOD,cudaMemcpyDeviceToHost);


            cublasDestroy_v2(cublas_handle);
        }

//        std::cout << "Computed L1 distances\n";


        cudaFree(corr_octave_gpu);
        cudaFree(corr_x_left_gpu);
        cudaFree(corr_x_right_gpu);
        cudaFree(corr_y_gpu);
        cudaFree(corr_distances_gpu);
        cudaFree(height_gpu);
        cudaFree(width_gpu);

        cudaFree(images_left_gpu);
        cudaFree(images_right_gpu);

        //        for(int i=0;i<im_left_gpu.size();i++)
        //        {
        //            cudaFree(im_left_gpu[i]);
        //            cudaFree(im_right_gpu[i]);
        //        }


        cudaFree(distance_L1_vector);
        cudaFree(distance_L1_gpu);
        cudaFree(ones_gpu);
    }


    {
        std::vector<std::pair<int, int> > vDistIdx;
        vDistIdx.reserve(mvKeys.size());


        mvuRight.resize(mvKeys.size(), -1.0f);
        mvDepth.resize(mvKeys.size(), -1.0f);

        for(int i=0;i<corr_match_left_idx.size();i++)
        {
            int bestDist = INT_MAX;
            int bestR = 0;

            int offset = i * NBRHOOD;

            for(int l=0; l< NBRHOOD; l++)
            {
                float dist = distance_L1[offset + l];
                if(dist < bestDist)
                {
                    bestDist = dist;
                    bestR = l;
                }
            }

            if(bestR== 0 || bestR== NBRHOOD-1)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = distance_L1[offset + bestR-1];
            const float dist2 = distance_L1[offset + bestR];
            const float dist3 = distance_L1[offset + bestR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;


            int left_idx = corr_match_left_idx[i];
            int right_idx = corr_match_right_idx[i];

            const cv::KeyPoint &kpL = mvKeys[left_idx];
            const cv::KeyPoint &kpR = mvKeysRight[right_idx];

            const float uL = kpL.pt.x;
            const float uR0 = kpR.pt.x;

            const float scaleFactor = inv_scale_[kpL.octave];
            const float scaleduR0 = round(uR0 * scaleFactor);

            // Re-scaled coordinate
            float bestuR = scale_[kpL.octave]*((float)scaleduR0+(float)bestR-NBRHOOD_HALF+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[left_idx]=mbf/disparity;
                mvuRight[left_idx] = bestuR;
                vDistIdx.push_back(std::pair<int,int>(bestDist, left_idx));
            }
        }


        sort(vDistIdx.begin(),vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size()/2].first;
        const float thDist = 1.5f*1.4f*median;

        for(int i=vDistIdx.size()-1;i>=0;i--)
        {
            if(vDistIdx[i].first<thDist)
                break;
            else
            {
                mvuRight[vDistIdx[i].second]=-1;
                mvDepth[vDistIdx[i].second]=-1;
            }
        }
    }
}


}

#endif
