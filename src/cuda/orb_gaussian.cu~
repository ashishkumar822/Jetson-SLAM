#include "cuda/orb_gpu.hpp"

#include<vector>
#include<opencv2/opencv.hpp>

#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<chrono>


namespace orb_cuda {


#define CUDA_NUM_THREADS_PER_BLOCK_V2 32

#define CUDA_NUM_THREADS_PER_BLOCK 512


__global__ void imgaussian_GPU(int n_threads,
                               int height, int width,
                               int roiheight, int roiwidth,
                               unsigned char* image_data,
                               int image_pitch,
                               unsigned char* image_data_gaussian,
                               int image_gaussian_pitch,
                               float* gaussian_weights)
{
    int index = blockDim.x * blockIdx.x +  threadIdx.x;

    if(index < n_threads)
    {
        int h = index / roiwidth;
        int w = index % roiwidth;

        int h_im = h + BORDER_SKIP;
        int w_im = w + BORDER_SKIP;

        int count = 0;
        float val = 0;

//#define GAUSSIAN_UNROLL

#ifndef GAUSSIAN_UNROLL

#pragma unroll
        for(int i=-3;i<=3;i++)
        {
            int row_offset = (h_im + i) * image_pitch;
#pragma unroll
            for(int j=-3;j<=3;j++)
            {
                int col_offset = w_im + j;
                val += gaussian_weights[count++] * image_data[row_offset + col_offset];
            }
        }

#else

        int offset = 0;

        offset = (h_im - 3) * image_pitch + w_im;

        val += gaussian_weights[0] * image_data[offset-3];
        val += gaussian_weights[1] * image_data[offset-2];
        val += gaussian_weights[2] * image_data[offset-1];
        val += gaussian_weights[3] * image_data[offset];
        val += gaussian_weights[4] * image_data[offset+1];
        val += gaussian_weights[5] * image_data[offset+2];
        val += gaussian_weights[6] * image_data[offset+3];

        offset = (h_im - 2) * image_pitch + w_im;

        val += gaussian_weights[7] * image_data[offset-3];
        val += gaussian_weights[8] * image_data[offset-2];
        val += gaussian_weights[9] * image_data[offset-1];
        val += gaussian_weights[10] * image_data[offset];
        val += gaussian_weights[11] * image_data[offset+1];
        val += gaussian_weights[12] * image_data[offset+2];
        val += gaussian_weights[13] * image_data[offset+3];


        offset = (h_im - 1) * image_pitch + w_im;

        val += gaussian_weights[14] * image_data[offset-3];
        val += gaussian_weights[15] * image_data[offset-2];
        val += gaussian_weights[16] * image_data[offset-1];
        val += gaussian_weights[17] * image_data[offset];
        val += gaussian_weights[18] * image_data[offset+1];
        val += gaussian_weights[19] * image_data[offset+2];
        val += gaussian_weights[20] * image_data[offset+3];

        offset = (h_im - 0) * image_pitch + w_im;

        val += gaussian_weights[21] * image_data[offset-3];
        val += gaussian_weights[22] * image_data[offset-2];
        val += gaussian_weights[23] * image_data[offset-1];
        val += gaussian_weights[24] * image_data[offset];
        val += gaussian_weights[25] * image_data[offset+1];
        val += gaussian_weights[26] * image_data[offset+2];
        val += gaussian_weights[27] * image_data[offset+3];

        offset = (h_im + 1) * image_pitch + w_im;

        val += gaussian_weights[28] * image_data[offset-3];
        val += gaussian_weights[29] * image_data[offset-2];
        val += gaussian_weights[30] * image_data[offset-1];
        val += gaussian_weights[31] * image_data[offset];
        val += gaussian_weights[32] * image_data[offset+1];
        val += gaussian_weights[33] * image_data[offset+2];
        val += gaussian_weights[34] * image_data[offset+3];

        offset = (h_im + 2) * image_pitch + w_im;

        val += gaussian_weights[35] * image_data[offset-3];
        val += gaussian_weights[36] * image_data[offset-2];
        val += gaussian_weights[37] * image_data[offset-1];
        val += gaussian_weights[38] * image_data[offset];
        val += gaussian_weights[39] * image_data[offset+1];
        val += gaussian_weights[40] * image_data[offset+2];
        val += gaussian_weights[41] * image_data[offset+3];

        offset = (h_im + 3) * image_pitch + w_im;

        val += gaussian_weights[42] * image_data[offset-3];
        val += gaussian_weights[43] * image_data[offset-2];
        val += gaussian_weights[44] * image_data[offset-1];
        val += gaussian_weights[45] * image_data[offset];
        val += gaussian_weights[46] * image_data[offset+1];
        val += gaussian_weights[47] * image_data[offset+2];
        val += gaussian_weights[48] * image_data[offset+3];
#endif

        image_data_gaussian[h_im * image_gaussian_pitch + w_im] = val;
    }

}



__global__ void imgaussian_GPU_v2(int n_threads,
                                  int height, int width,
                                  int roiheight, int roiwidth,
                                  unsigned char* image_data,
                                  unsigned char* image_data_gaussian,
                                  float* gaussian_weights)
{
    int index = blockDim.x * blockIdx.x +  threadIdx.x;

    if(index < n_threads)
    {
        if(threadIdx.x < 28)   // per block only 3 locations
        {
            __shared__ float shared_sum[32];

            const int idx = (threadIdx.x / 7) +  3 * blockIdx.x;
            const int idx_w = (threadIdx.x % 7);

            const int idx_h = (threadIdx.x / 7);

            //            const int eff_h = height - 2*BORDER_SKIP;
            const int eff_w = width - 2*3;

            int h = idx / eff_w;
            int w = idx % eff_w;

            int h_im = h + 3;
            int w_im = w + 3;

            if( h_im < height && w_im < width)
            {
                int count =0;
                int row_offset = 0;
                int col_offset = 0;

                col_offset =  w_im - 3 + idx_w;

                float sum = 0;
                for(int i=-3;i<3;i++)
                {
                    row_offset = (h_im + i) * width;

                    sum += gaussian_weights[count*7+idx_w] * image_data[row_offset + col_offset];
                    count++;
                }

                shared_sum[threadIdx.x] = sum;

                __syncthreads();

                if(threadIdx.x % 7 == 0)
                {
                    float sum = 0;
                    for(int i=0;i<7;i++)
                        sum += shared_sum[idx_h*7+i];

                    image_data_gaussian[h_im * width + w_im] = sum;

                }
            }

        }
    }
}



void ORB_GPU::compute_gaussian(int height, int width,
                               unsigned char* image_data_gpu,
                               int image_pitch,
                               unsigned char* image_data_gaussian_gpu,
                               int image_gaussian_pitch,
                               float* gaussian_weights_gpu,
                               cudaStream_t& cuda_stream)
{
    {
        // 3 is (7-1)/2 for gaussian kernel of 7
        // we dont process border because they are not involved in anycomputation
        // in orb because of BORDER_SKIP and BORDER_OFFSET
        int roiheight = (height - 2 * BORDER_SKIP);
        int roiwidth  = (width - 2 * BORDER_SKIP);

        int n_threads =  roiheight * roiwidth;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK  + CUDA_NUM_THREADS_PER_BLOCK;


        imgaussian_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                          n_threads,
                                                                                          height, width,
                                                                                          roiheight, roiwidth,
                                                                                          image_data_gpu,
                                                                                          image_pitch,
                                                                                          image_data_gaussian_gpu,
                                                                                          image_gaussian_pitch,
                                                                                          gaussian_weights_gpu);


        // cz 3 per block

        //        int CUDA_NUM_BLOCKS = ((roiheight * roiwidth - 1) / 3)  + 1;
        //        int n_threads =  CUDA_NUM_BLOCKS * CUDA_NUM_THREADS_PER_BLOCK_V2;


        //        imgaussian_GPU_v2<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK_V2, 0, cuda_stream>>>(
        //                                                                                             n_threads,
        //                                                                                             height, width,
        //                                                                                             roiheight, roiwidth,
        //                                                                                             image_data_gpu,
        //                                                                                             image_data_gaussian_gpu,
        //                                                                                             gaussian_weights_gpu);

    }
}




__global__ void imgaussian_GPU_unroll(int n_threads,
                                      int height, int width,
                                      int roiheight, int roiwidth,
                                      unsigned char* image_data,
                                      float* gaussian_weights,
                                      unsigned char* gaussian_unrolling)
{
    int index = blockDim.x * blockIdx.x +  threadIdx.x;

    if(index < n_threads)
    {
        int h = (index / 7) / roiwidth;
        int w = (index / 7) % roiwidth;

        int h_im = h + 3;
        int w_im = w + 3;

        int kernel_loc_w = index % 7;
        int col_offset = w_im + kernel_loc_w - 3;

        float val = 0;
        int count = 0;

        for(int i=-3;i<=3;i++)
        {
            int row_offset = (h_im + i) * width;
            val += gaussian_weights[count * 7 + kernel_loc_w] * image_data[row_offset + col_offset];
            count++;
        }

        const int spatial_offset = roiwidth * roiheight;
        const int spatial_w = index / 7;

        gaussian_unrolling[kernel_loc_w * spatial_offset + spatial_w] = val;
    }

}


__global__ void imgaussian_GPU_reduce(int n_threads,
                                      int height, int width,
                                      int roiheight, int roiwidth,
                                      unsigned char* gaussian_unrolling,
                                      unsigned char* image_data_gaussian)
{
    int index = blockDim.x * blockIdx.x +  threadIdx.x;

    if(index < n_threads)
    {
        const int h = index / roiwidth;
        const int w = index % roiwidth;

        const int h_im = h + 3;
        const int w_im = w + 3;

        const int col_offset = index;

        float val = 0;

        for(int i=0;i<7;i++)
        {
            int row_offset = i * n_threads;
            val +=  gaussian_unrolling[row_offset + col_offset];
        }

        image_data_gaussian[h_im * width + w_im] = val;
    }
}


//__global__ void imgaussian_GPU_unroll(int n_threads,
//                                      int height, int width,
//                                      int roiheight, int roiwidth,
//                                      unsigned char* image_data,
//                                      float* gaussian_weights,
//                                      unsigned char* gaussian_unrolling)
//{
//    int index = blockDim.x * blockIdx.x +  threadIdx.x;

//    if(index < n_threads)
//    {
//        int h = (index / 7) / roiwidth;
//        int w = (index / 7) % roiwidth;

//        int h_im = h + 3;
//        int w_im = w + 3;

//        int kernel_loc_w = index % 7;
//        int col_offset = w_im + kernel_loc_w - 3;

//        float val = 0;
//        int count = 0;

//        for(int i=-3;i<=3;i++)
//        {
//            int row_offset = (h_im + i) * width;
//            val += gaussian_weights[count * 7 + kernel_loc_w] * image_data[row_offset + col_offset];
//            count++;
//        }

//        const int idx = h * roiwidth + w;

//        gaussian_unrolling[idx * 8 + kernel_loc_w] = val;
//    }

//}


//__global__ void imgaussian_GPU_reduce(int n_threads,
//                                      int height, int width,
//                                      int roiheight, int roiwidth,
//                                      unsigned char* gaussian_unrolling,
//                                      unsigned char* image_data_gaussian)
//{
//    int index = blockDim.x * blockIdx.x +  threadIdx.x;

//    if(index < n_threads)
//    {
//        __shared__ int shared_sum[32];

//        const int thread_blkidx = threadIdx.x;

//        const int h = (index/32) / roiwidth;
//        const int w = (index/32) % roiwidth;

//        const int h_im = h + 3;
//        const int w_im = w + 3;

//        shared_sum[thread_blkidx] = gaussian_unrolling[index];

//        int groupsize = 4;

//        for(int i=0;i<3;i++)
//        {
//            if(thread_blkidx < groupsize)
//                shared_sum[thread_blkidx] += shared_sum[thread_blkidx + groupsize];
//        }

//        if(thread_blkidx == 0)
//        {
//            image_data_gaussian[h_im * width + w_im] = shared_sum[0];
//        }

//    }
//}



void ORB_GPU::compute_gaussian_v2(int height, int width,
                                  unsigned char* image_data_gpu,
                                  unsigned char* gaussian_unrolling,
                                  unsigned char* image_data_gaussian_gpu,
                                  float* gaussian_weights_gpu,
                                  cudaStream_t& cuda_stream)
{


    {
        // 3 is (7-1)/2 for gaussian kernel of 7
        // we dont process border because they are not involved in anycomputation
        // in orb because of BORDER_SKIP and BORDER_OFFSET
        int roiheight = (height - 2 * 3);
        int roiwidth  = (width - 2 * 3);

        int n_threads =  roiheight * roiwidth * 7;

        int CUDA_NUM_BLOCKS = (n_threads -1 ) / CUDA_NUM_THREADS_PER_BLOCK + 1;


        imgaussian_GPU_unroll<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                 n_threads,
                                                                                                 height, width,
                                                                                                 roiheight, roiwidth,
                                                                                                 image_data_gpu,
                                                                                                 gaussian_weights_gpu,
                                                                                                 gaussian_unrolling);

        cudaStreamSynchronize(cuda_stream);
        std::cout << " ---- " << cudaGetErrorString(cudaGetLastError()) << "\n";

    }


    {
        // 3 is (7-1)/2 for gaussian kernel of 7
        // we dont process border because they are not involved in anycomputation
        // in orb because of BORDER_SKIP and BORDER_OFFSET
        int roiheight = (height - 2 * 3);
        int roiwidth  = (width - 2 * 3);

        int n_threads =  roiheight * roiwidth;

        int CUDA_NUM_BLOCKS = (n_threads -1 ) / CUDA_NUM_THREADS_PER_BLOCK + 1;


        imgaussian_GPU_reduce<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                 n_threads,
                                                                                                 height, width,
                                                                                                 roiheight, roiwidth,
                                                                                                 gaussian_unrolling,
                                                                                                 image_data_gaussian_gpu);

        cudaStreamSynchronize(cuda_stream);

        std::cout << " **** " << cudaGetErrorString(cudaGetLastError()) << "\n";
    }


    //    {
    //        // 3 is (7-1)/2 for gaussian kernel of 7
    //        // we dont process border because they are not involved in anycomputation
    //        // in orb because of BORDER_SKIP and BORDER_OFFSET
    //        int roiheight = (height - 2 * 3);
    //        int roiwidth  = (width - 2 * 3);

    //        int n_threads =  roiheight * roiwidth * 7;

    //        int CUDA_NUM_BLOCKS = (n_threads -1 ) / CUDA_NUM_THREADS_PER_BLOCK + 1;


    //        imgaussian_GPU_unroll<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
    //                                                                                                 n_threads,
    //                                                                                                 height, width,
    //                                                                                                 roiheight, roiwidth,
    //                                                                                                 image_data_gpu,
    //                                                                                                 gaussian_weights_gpu,
    //                                                                                                 gaussian_unrolling);

    //        cudaStreamSynchronize(cuda_stream);
    //        std::cout << " ---- " << cudaGetErrorString(cudaGetLastError()) << "\n";

    //    }


    //    {
    //        // 3 is (7-1)/2 for gaussian kernel of 7
    //        // we dont process border because they are not involved in anycomputation
    //        // in orb because of BORDER_SKIP and BORDER_OFFSET
    //        int roiheight = (height - 2 * 3);
    //        int roiwidth  = (width - 2 * 3);

    //        int n_threads =  roiheight * roiwidth * 32;

    //        int CUDA_NUM_BLOCKS = (n_threads -1 ) / CUDA_NUM_THREADS_PER_BLOCK_V2 + 1;


    //        imgaussian_GPU_reduce<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK_V2, 0, cuda_stream>>>(
    //                                                                                                 n_threads,
    //                                                                                                 height, width,
    //                                                                                                 roiheight, roiwidth,
    //                                                                                                 gaussian_unrolling,
    //                                                                                                 image_data_gaussian_gpu);

    //        cudaStreamSynchronize(cuda_stream);

    //        std::cout << " **** " << cudaGetErrorString(cudaGetLastError()) << "\n";
    //    }



}

}
