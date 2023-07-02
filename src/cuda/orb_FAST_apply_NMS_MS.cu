#include "cuda/orb_gpu.hpp"

#include<chrono>
#include<tictoc.hpp>

using namespace std;

namespace orb_cuda {

#define CUDA_NUM_THREADS_PER_BLOCK 32

#define NMS_WINDOW 3
#define NMS_WINDOW_HALF (NMS_WINDOW - 1 ) / 2

#define DIVERGENCE_LESS_CODE


__global__ void Fill_s0_score_kernel(int n_threads,
                                     int imheight, int imwidth,
                                     int* kp_x,
                                     int* kp_y,
                                     int* kp_score,
                                     int* kp_level,
                                     float* kp_scale_factor,
                                     int* s0_score_gpu)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int score = kp_score[thread_idx];

        if(score)
        {
            //            int h = rintf(kp_y[thread_idx] * scale);
            //            int w = rintf(kp_x[thread_idx] * scale);

            const float scale = kp_scale_factor[thread_idx];
            const int level = kp_level[thread_idx];

            int h = kp_y[thread_idx] * scale;
            int w = kp_x[thread_idx] * scale;

            s0_score_gpu[(level * imheight + h) * imwidth + w] = score;
        }
    }
}






//__global__ void NMS_S_s0_score_kernel(int n_threads,
//                                      int imheight, int imwidth,
//                                      int n_levels,
//                                      int* kp_x,
//                                      int* kp_y,
//                                      int* kp_score,
//                                      int* kp_level,
//                                      float* kp_scale_factor,
//                                      int* s0_score_gpu,
//                                      int* nms_s_score_gpu,
//                                      int* nms_s_level_gpu)
//{
//    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

//    if(thread_idx <  n_threads)
//    {
//        int score = kp_score[thread_idx];

//        if(score)
//        {
//            //            int h = rintf(kp_y[thread_idx] * scale);
//            //            int w = rintf(kp_x[thread_idx] * scale);

//            const float scale = kp_scale_factor[thread_idx];
//            const int level = kp_level[thread_idx];

//            int h = kp_y[thread_idx] * scale;
//            int w = kp_x[thread_idx] * scale;

//            int max_score = 0;
//            int max_level = -1;

//            for(int i=0;i<n_levels;i++)
//            {
//                int score = s0_score_gpu[(i * imheight + h ) * imwidth + w];

//                // divergence code
//                if(score >  max_score)
//                {
//                    max_score = score;
//                    max_level = i;
//                }

//                //                //divergence less code
//                //                int res = score >  max_score;
//                //                int not_res = 1 - res;

//                //                max_score = res * score + not_res * score;
//                //                max_level = res * i + not_res * max_level;
//            }


//            if(level == max_level)
//            {
//                nms_s_score_gpu[h * imwidth + w] = max_score;
//                nms_s_level_gpu[h * imwidth + w] = max_level;
//            }
//            else
//            {
//                kp_score[thread_idx] = 0;
//            }

//            s0_score_gpu[(level * imheight + h ) * imwidth + w] = 0;
//        }
//    }
//}




//__global__ void NMS_L_s0_score_kernel(int n_threads,
//                                      int imheight, int imwidth,
//                                      int* kp_x,
//                                      int* kp_y,
//                                      int* kp_score,
//                                      int* kp_level,
//                                      float* kp_scale_factor,
//                                      int* nms_s_score_gpu,
//                                      int* nms_s_level_gpu)
//{
//    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

//    if(thread_idx <  n_threads)
//    {
//        int score = kp_score[thread_idx];

//        if(score)
//        {
//            //            int h = rintf(kp_y[thread_idx] * scale);
//            //            int w = rintf(kp_x[thread_idx] * scale);

//            const float scale = kp_scale_factor[thread_idx];
//            const int level = kp_level[thread_idx];

//            int h_im = kp_y[thread_idx] * scale;
//            int w_im = kp_x[thread_idx] * scale;

//            int nms_score = nms_s_score_gpu[h_im * imwidth + w_im];
//            int nms_level = nms_s_level_gpu[h_im * imwidth + w_im];

//            int valid = 1;

//            if(nms_level == level)
//            {
//#if(NMS_WINDOW==3)
//                //unrolled for 3x3
//                int row_offset = (h_im + -1) * imwidth;
//                int col_offset = w_im + -1;
//                int total_offset = row_offset + col_offset;

//                valid *= nms_score >= nms_s_score_gpu[total_offset];
//                valid *= nms_score >= nms_s_score_gpu[total_offset+1];
//                valid *= nms_score >= nms_s_score_gpu[total_offset+2];

//                row_offset = (h_im + 0) * imwidth;
//                total_offset = row_offset + col_offset;

//                valid *= nms_score >= nms_s_score_gpu[total_offset];
//                valid *= nms_score >= nms_s_score_gpu[total_offset+1];
//                valid *= nms_score >= nms_s_score_gpu[total_offset+2];


//                row_offset = (h_im + 1) * imwidth;
//                total_offset = row_offset + col_offset;

//                valid *= nms_score >= nms_s_score_gpu[total_offset];
//                valid *= nms_score >= nms_s_score_gpu[total_offset+1];
//                valid *= nms_score >= nms_s_score_gpu[total_offset+2];

//#else

//#ifdef DIVERGENCE_LESS_CODE

//                //no divergence
//                for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
//                {
//                    int row_offset = (h_im + i) * imwidth;

//                    for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
//                    {
//                        int col_offset = w_im + j;

//                        int res = (nms_score >= nms_s_score_gpu[row_offset + col_offset]);
//                        valid *= (res);
//                    }
//                }

//#else
//                // divergence
//                for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
//                {
//                    int row_offset = (h_im + i) * imwidth;

//                    for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
//                    {
//                        int col_offset = w_im + j;

//                        if(nms_score < nms_s_score_gpu[row_offset + col_offset])
//                        {
//                            valid = 0;
//                            break;
//                        }
//                    }

//                    if(!valid)break;
//                }
//#endif

//#endif
//            }
//            else
//                valid = 0;

//            kp_score[thread_idx] = kp_score[thread_idx] * valid;
//        }
//    }
//}




__global__ void NMS_S_s0_score_kernel(int n_threads,
                                      int imheight, int imwidth,
                                      int n_levels,
                                      int* kp_x,
                                      int* kp_y,
                                      int* kp_score,
                                      int* kp_level,
                                      float* kp_scale_factor,
                                      int* s0_score_gpu,
                                      int* nms_s_score_gpu,
                                      int* nms_s_level_gpu)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int score = kp_score[thread_idx];

        if(score)
        {
            //            int h = rintf(kp_y[thread_idx] * scale);
            //            int w = rintf(kp_x[thread_idx] * scale);

            const float scale = kp_scale_factor[thread_idx];
            const int level = kp_level[thread_idx];

            int h = kp_y[thread_idx] * scale;
            int w = kp_x[thread_idx] * scale;

            int max_score = 0;
            int max_level = 0;

            int sum = 0;
            int sum_level = 0;

            for(int i=0;i<n_levels;i++)
            {
//                printf("%d %d %d\n", (i * imheight + h ) * imwidth + w, h, w);
                int score = s0_score_gpu[(i * imheight + h ) * imwidth + w];

                // divergence code
                if(score >  max_score)
                {
                    max_score = score;
                    max_level = i;
                }

                sum+= score;
                if(!score)
                    sum_level++;
                //                //divergence less code
                //                int res = score >  max_score;
                //                int not_res = 1 - res;

                //                max_score = res * score + not_res * score;
                //                max_level = res * i + not_res * max_level;
            }

            if(level == max_level)
            {
                //                nms_s_score_gpu[h * imwidth + w] = max_score;
                //                nms_s_level_gpu[h * imwidth + w] = max_level;
                nms_s_score_gpu[h * imwidth + w] = sum;
                nms_s_level_gpu[h * imwidth + w] = sum_level;

            }
//            else
//            {
//                kp_score[thread_idx] = 0;
//            }

            s0_score_gpu[(level * imheight + h ) * imwidth + w] = 0;
        }
    }
}




__global__ void NMS_L_s0_score_kernel(int n_threads,
                                      int imheight, int imwidth,
                                      int* kp_x,
                                      int* kp_y,
                                      int* kp_score,
                                      int* kp_level,
                                      float* kp_scale_factor,
                                      int* nms_s_score_gpu,
                                      int* nms_s_level_gpu)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int score = kp_score[thread_idx];

        if(score)
        {
            //            int h = rintf(kp_y[thread_idx] * scale);
            //            int w = rintf(kp_x[thread_idx] * scale);

            const float scale = kp_scale_factor[thread_idx];
            const int level = kp_level[thread_idx];

            int h_im = kp_y[thread_idx] * scale;
            int w_im = kp_x[thread_idx] * scale;

            int nms_score = nms_s_score_gpu[h_im * imwidth + w_im];
            int nms_level = nms_s_level_gpu[h_im * imwidth + w_im];

            int valid = 1;

            //                        if(nms_level == level)
            {
#pragma unroll
                for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
                {
                    //                    if(i > -2 && i < 2)
                    //                        continue;

                    int row_offset = (h_im + i) * imwidth;

#pragma unroll
                    for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                    {
                        //                        if(j > -2 && j < 2)
                        //                            continue;

                        int col_offset = w_im + j;

                        int nbr_score = nms_s_score_gpu[row_offset + col_offset];
                        int nbr_level = nms_s_level_gpu[row_offset + col_offset];

                        //                        if(nms_score < nbr_score)
                        //                        {
                        //                            if(nms_level < nbr_level)
                        //                            {
                        //                                valid *=0;
                        //                            }
                        //                        }
                        //                        else

                        //                        int res = (nms_score >= nms_s_score_gpu[row_offset + col_offset]);

                        valid &= (nms_score * nms_level) >= (nbr_score * nbr_level);
                    }
                }
            }

            kp_score[thread_idx] = valid ? kp_score[thread_idx] : 0;
        }
    }
}


void ORB_GPU::apply_NMS_MS_S_L(int height, int width,
                               int n_levels,
                               int n_keypoints,
                               int* keypoints_x,
                               int* keypoints_y,
                               int* keypoints_score,
                               int* keypoints_level,
                               float* keypoints_scale_factor,
                               int* s0_score_gpu,
                               int *nms_s_score,
                               int *nms_s_level,
                               cudaStream_t& cuda_stream)
{

    // fill s0_score
    {

        int n_threads =  n_keypoints;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1;

        Fill_s0_score_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                n_threads,
                                                                                                height, width,
                                                                                                keypoints_x,
                                                                                                keypoints_y,
                                                                                                keypoints_score,
                                                                                                keypoints_level,
                                                                                                keypoints_scale_factor,
                                                                                                s0_score_gpu);
    }

//    cudaStreamSynchronize(cuda_stream);
//    std::cout << "s0 = " << cudaGetErrorString(cudaGetLastError()) << "\n";

    // NMS-s
    {

        int n_threads =  n_keypoints;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1;


        NMS_S_s0_score_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                 n_threads,
                                                                                                 height, width,
                                                                                                 n_levels,
                                                                                                 keypoints_x,
                                                                                                 keypoints_y,
                                                                                                 keypoints_score,
                                                                                                 keypoints_level,
                                                                                                 keypoints_scale_factor,
                                                                                                 s0_score_gpu,
                                                                                                 nms_s_score,
                                                                                                 nms_s_level);

    }


    // NMS-l
    {

        int n_threads =  n_keypoints;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1;

        NMS_L_s0_score_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                 n_threads,
                                                                                                 height, width,
                                                                                                 keypoints_x,
                                                                                                 keypoints_y,
                                                                                                 keypoints_score,
                                                                                                 keypoints_level,
                                                                                                 keypoints_scale_factor,
                                                                                                 nms_s_score,
                                                                                                 nms_s_level);

    }
}

}
