#include "cuda/orb_gpu.hpp"

#include<chrono>


namespace orb_cuda {


#define CUDA_NUM_THREADS_PER_BLOCK 512


__global__ void ORB_copy_output_GPU(int n_threads,
                                    const int octave,
                                    const int width,
                                    const float scale,
                                    const int* keypoints_x,
                                    const int* keypoints_y,
                                    const int* keypoints_score,
                                    const float* keypoints_angle,
                                    int* keypoints_x_op,
                                    int* keypoints_y_op,
                                    float* keypoints_angle_op,
                                    int* keypoints_response_op,
                                    int* keypoints_octave_op,
                                    int* keypoints_size_op)
{
    int index = blockDim.x * blockIdx.x +  threadIdx.x;

    if(index < n_threads)
    {
        const int kp_x =keypoints_x[index];
        const int kp_y =keypoints_y[index];

        keypoints_x_op[index] = kp_x * scale;
        keypoints_y_op[index] = kp_y * scale;
        keypoints_angle_op[index] = keypoints_angle[index] * (180.0 / M_PI);

//        const int offset = kp_y * width + kp_x;
        keypoints_response_op[index] = keypoints_score[index];

        keypoints_octave_op[index] = octave;
        keypoints_size_op[index] = 31 * scale;
    }

}



void ORB_GPU::copy_output(const int n_keypoints,
                          const int octave,
                          const int width,
                          const float scale,
                          const int* keypoints_x,
                          const int* keypoints_y,
                          const int* keypoints_score,
                          const float* keypoints_angle,
                          int* keypoints_x_op,
                          int* keypoints_y_op,
                          int* keypoints_response_op,
                          float* keypoints_angle_op,
                          int* keypoints_octave_op,
                          int* keypoints_size_op,
                          cudaStream_t& cuda_stream)
{

    const int n_threads =  n_keypoints;

    int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK  + CUDA_NUM_THREADS_PER_BLOCK;


    ORB_copy_output_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                           n_threads,
                                                                                           octave,
                                                                                           width,
                                                                                           scale,
                                                                                           keypoints_x,
                                                                                           keypoints_y,
                                                                                           keypoints_score,
                                                                                           keypoints_angle,
                                                                                           keypoints_x_op,
                                                                                           keypoints_y_op,
                                                                                           keypoints_angle_op,
                                                                                           keypoints_response_op,
                                                                                           keypoints_octave_op,
                                                                                           keypoints_size_op
                                                                                           );


}

}
