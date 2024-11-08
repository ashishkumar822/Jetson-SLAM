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


#include<cuda/orb_gpu.hpp>

#include<chrono>

namespace orb_cuda {


#define CUDA_NUM_THREADS_PER_BLOCK 32


//#define PATCH_SIZE  31
#define HALF_PATCH_SIZE  15
//#define EDGE_THRESHOLD  19



__global__ void FASTComputeOrientationGPU(int n_threads,
                                          int imheight, int imwidth,
                                          int* u_max,
                                          unsigned char* image_data,
                                          int image_pitch,
                                          int* keypoint_x,
                                          int* keypoint_y,
                                          int* keypoints_score,
                                          float* keypoint_angle)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        if(keypoints_score[index])
        {
            int h_im = keypoint_y[index];
            int w_im = keypoint_x[index];

            int offset = h_im * image_pitch + w_im;

            const unsigned char* center = image_data + offset;

            int m_01 = 0, m_10 = 0;

            for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
                m_10 += u * center[u];

            const int step = image_pitch;
            for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
            {
                int v_sum = 0;
                int d = u_max[v];

                for (int u = -d; u <= d; ++u)
                {
                    int val_plus =  center[u + v*step];
                    int val_minus = center[u - v*step];
                    v_sum += (val_plus - val_minus);
                    m_10 += u * (val_plus + val_minus);
                }
                m_01 += v * v_sum;
            }

            keypoint_angle[index] = atan2f((float)m_01, (float)m_10);
        }

    }
}



__global__ void FASTComputeOrientationGPUv2(int n_threads,
                                            int imheight, int imwidth,
                                            int* u_max,
                                            unsigned char* image_data,
                                            int image_pitch,
                                            int* keypoint_x,
                                            int* keypoint_y,
                                            int* keypoints_score,
                                            float* keypoint_angle)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        if(keypoints_score[index])
        {

            __shared__ int shared_m10[32];
            __shared__ int shared_m01[32];

            const int wcell = 32;
            //        const int hcell = 1;
            const int whcell = 16;

            int cell_index = index / 32;

            int h_im = keypoint_y[index];
            int w_im = keypoint_x[index];


            int w_patch = (index % wcell) - whcell;

            int v_plus = 0;
            int v_minus = 0;

            for(int i=0;i<16;i++)
            {
                int val1;
                int val2;

                {
                    int h_patch = i;
                    int offset = (h_im+h_patch) * image_pitch + (w_im+w_patch);

                    const unsigned char* center = image_data + offset;
                    val1 = center[0];
                }

                {
                    int h_patch = -i-1;
                    int offset = (h_im+h_patch) * image_pitch + (w_im+w_patch);

                    const unsigned char* center = image_data + offset;
                    val2 = center[0];
                }

                v_plus += val1 + val2;
                v_minus += i * val1 - (-i-1) * val2;

            }

            int u = threadIdx.x;

            shared_m10[u] = u * v_plus;
            shared_m01[u] = v_minus;

            __syncthreads();

            if(index==0)
            {
                int m_10 = 0;
                int m_01 = 0;

                for(int i=0;i<32;i++)
                {
                    m_10 += shared_m10[i];
                    m_01 += shared_m01[i];
                }
                keypoint_angle[cell_index] = atan2f((float)m_01, (float)m_10);
            }
        }
    }
}



__global__ void FASTComputeOrientationGPUv3(int n_threads,
                                            int imheight, int imwidth,
                                            int* u_max,
                                            unsigned char* image_data,
                                            int image_pitch,
                                            int* keypoint_x,
                                            int* keypoint_y,
                                            int* keypoints_score,
                                            float* keypoint_angle)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        if(keypoints_score[index])
        {

            __shared__ int shared_m10[31];
            __shared__ int shared_m01[31];

            const int wcell = 32;
            //        const int hcell = 1;
            const int whcell = 15;

            int cell_index = index / 32;

            int h_im = keypoint_y[index];
            int w_im = keypoint_x[index];


            int my_id = (index % wcell);

            int w_patch = my_id - whcell;

            int v_plus = 0;
            int v_minus = 0;

            if(my_id < 31)
            {
                for(int i=0;i<16;i++)
                {
                    int val1;
                    int val2;

                    if(i==0)
                    {
                        {
                            int h_patch = i;
                            int offset = (h_im+h_patch) * image_pitch + (w_im+w_patch);

                            const unsigned char* center = image_data + offset;
                            val1 = center[0];
                        }
                        v_plus += val1 ;
                    }
                    else
                    {
                        {
                            int h_patch = i;
                            int offset = (h_im+h_patch) * image_pitch + (w_im+w_patch);

                            const unsigned char* center = image_data + offset;
                            val1 = center[0];
                        }

                        {
                            int h_patch = -i;
                            int offset = (h_im+h_patch) * image_pitch + (w_im+w_patch);

                            const unsigned char* center = image_data + offset;
                            val2 = center[0];
                        }

                        v_plus += val1 + val2;
                        v_minus += i * (val1 - val2);
                    }

                }

                shared_m10[my_id] = w_patch * v_plus;
                shared_m01[my_id] = v_minus;
            }

            __syncthreads();

            if(index==0)
            {
                int m_10 = 0;
                int m_01 = 0;

                for(int i=0;i<31;i++)
                {
                    m_10 += shared_m10[i];
                    m_01 += shared_m01[i];
                }
                keypoint_angle[cell_index] = atan2f((float)m_01, (float)m_10);
            }
        }
    }
}





void ORB_GPU::FAST_compute_orientation(int height, int width,
                                       unsigned char* image_data_gpu,
                                       int image_pitch,
                                       int  n_keypoints,
                                       int* keypoints_x_gpu,
                                       int* keypoints_y_gpu,
                                       int* keypoints_score,
                                       float* keypoints_angle_gpu,
                                       int* umax_gpu,
                                       cudaStream_t& cuda_stream)
{

    int method = 0;//original
    //    int method = 1;//v2 i.e. shared mem, window = 32x32  // according to warp size of GPU to encourage memory coalescing
    //    int method = 2;//v2 i.e. shared mem, window = 32x32  // according to warp size of GPU to encourage memory coalescing


    // Original
    if(method == 0)
    {
        int n_threads = n_keypoints;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK  + CUDA_NUM_THREADS_PER_BLOCK;

        FASTComputeOrientationGPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                     n_threads,
                                                                                                     height, width,
                                                                                                     umax_gpu,
                                                                                                     image_data_gpu,
                                                                                                     image_pitch,
                                                                                                     keypoints_x_gpu,
                                                                                                     keypoints_y_gpu,
                                                                                                     keypoints_score,
                                                                                                     keypoints_angle_gpu
                                                                                                     );

//        cudaStreamSynchronize(cuda_stream);
//        std::cout << cudaGetErrorString(cudaGetLastError()) << " --\n";

    }
    else if(method == 1)
    {
        int n_threads = n_keypoints * 32;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK  + CUDA_NUM_THREADS_PER_BLOCK;

        FASTComputeOrientationGPUv2<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                       n_threads,
                                                                                                       height, width,
                                                                                                       umax_gpu,
                                                                                                       image_data_gpu,
                                                                                                       image_pitch,
                                                                                                       keypoints_x_gpu,
                                                                                                       keypoints_y_gpu,
                                                                                                       keypoints_score,
                                                                                                       keypoints_angle_gpu
                                                                                                       );
    }
    else if(method == 2)
    {
        int n_threads = n_keypoints * 32;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK  + CUDA_NUM_THREADS_PER_BLOCK;

        FASTComputeOrientationGPUv3<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                       n_threads,
                                                                                                       height, width,
                                                                                                       umax_gpu,
                                                                                                       image_data_gpu,
                                                                                                       image_pitch,
                                                                                                       keypoints_x_gpu,
                                                                                                       keypoints_y_gpu,
                                                                                                       keypoints_score,
                                                                                                       keypoints_angle_gpu
                                                                                                       );
    }
}

}
