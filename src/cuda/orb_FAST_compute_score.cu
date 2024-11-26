#include "cuda/orb_gpu.hpp"

#include<chrono>

namespace orb_cuda {

//15.84us at 1235x375
//#define CUDA_NUM_THREADS_PER_BLOCK_x 32 // to ensure memory coalescing 16x16 can be an option however half of the warp will access different mem while anlther half the other
//#define CUDA_NUM_THREADS_PER_BLOCK_y 8

//14.04us at 1235x375
#define CUDA_NUM_THREADS_PER_BLOCK_x 32 // to ensure memory coalescing 16x16 can be an option however half of the warp will access different mem while anlther half the other
#define CUDA_NUM_THREADS_PER_BLOCK_y 8


#define CUDA_NUM_THREADS_PER_BLOCK 512


#define th_FAST_MIN 9 // 0.6
#define th_FAST_MAX 13 // 0.9

#define GRID_LAUNCH

void ORB_GPU::patternCircle(int* pattern_circle, int rowStride, int patternSize)
{
    //    {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
    //    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}

    if(patternSize==16)
    {
        pattern_circle[0]  =  3*rowStride+0;
        pattern_circle[1]  =  3*rowStride+1;
        pattern_circle[2]  =  2*rowStride+2;
        pattern_circle[3]  =  1*rowStride+3;
        pattern_circle[4]  =  0*rowStride+3;
        pattern_circle[5]  = -1*rowStride+3;
        pattern_circle[6]  = -2*rowStride+2;
        pattern_circle[7]  = -3*rowStride+1;
        pattern_circle[8]  = -3*rowStride+0;
        pattern_circle[9]  = -3*rowStride-1;
        pattern_circle[10] = -2*rowStride-2;
        pattern_circle[11] = -1*rowStride-3;
        pattern_circle[12] =  0*rowStride-3;
        pattern_circle[13] =  1*rowStride-3;
        pattern_circle[14] =  2*rowStride-2;
        pattern_circle[15] =  3*rowStride-1;
    }
}



__global__ void FASTComputeScoreGPU_patternSize_16_v1(int n_threads,
                                                      int imheight, int imwidth,
                                                      int roiheight, int roiwidth,
                                                      int threshold,
                                                      const unsigned char* __restrict__ image_data,
                                                      int* __restrict__ score_data)
{

#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if(w < imwidth && h < imheight)
    {
        const int h_im = h;// + BORDER_SKIP;
        const int w_im = w;// + BORDER_SKIP;

        const int offset = h_im * imwidth + w_im;

        score_data[offset] =  0;

        if(w >= BORDER_SKIP && w < imwidth - BORDER_SKIP && h < imheight - BORDER_SKIP && h >= BORDER_SKIP)
        {

#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int h = index / roiwidth;
        int w = index % roiwidth;
#endif


        float local_score = 0;

        const unsigned char* ptr = image_data + offset;

        const int v = ptr[0];

        const int vt = v + threshold;
        const int v_t = v - threshold;

        int n_bright = 0;
        int n_dark = 0;

        const int pixel_circle_0  =  3*imwidth+0;
        const int pixel_circle_8  = -3*imwidth+0;

        const int ptr_0 = ptr[pixel_circle_0];
        const int ptr_8 = ptr[pixel_circle_8];

        if((ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t))
            return;


        n_bright = (ptr_0 > v_t) + (ptr_8 > vt);
        n_dark  = (ptr_0 < v_t) + (ptr_8 < vt);;


        const int pixel_circle_4  =  0*imwidth+3;
        const int pixel_circle_12 =  0*imwidth-3;
        const int ptr_4  = ptr[pixel_circle_4];
        const int ptr_12 = ptr[pixel_circle_12];


        if((ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t))
            return;

        n_bright += (ptr_4 > v_t) + (ptr_12 > vt);
        n_dark  += (ptr_4 < v_t) + (ptr_12 < vt);;


        const int pixel_circle_6  = -2*imwidth+2;
        const int pixel_circle_14 =  2*imwidth-2;

        const int ptr_6  = ptr[pixel_circle_6];
        const int ptr_14 = ptr[pixel_circle_14];


        if((ptr_6 <= vt && ptr_6 >= v_t && ptr_14 <= vt && ptr_14 >= v_t))
            return;

        n_bright += (ptr_6 > v_t) + (ptr_14 > vt);
        n_dark  += (ptr_6 < v_t) + (ptr_14 < vt);;


        const int pixel_circle_2  =  2*imwidth+2;
        const int pixel_circle_10 = -2*imwidth-2;
        const int ptr_2  = ptr[pixel_circle_2];
        const int ptr_10 = ptr[pixel_circle_10];


        if((ptr_2 <= vt && ptr_2 >= v_t && ptr_10 <= vt && ptr_10 >= v_t))
            return;

        n_bright += (ptr_2 > v_t) + (ptr_10 > vt);
        n_dark  += (ptr_2 < v_t) + (ptr_10 < vt);;


        const int pixel_circle_1   =  3*imwidth+1;
        const int pixel_circle_9   = -3*imwidth-1;

        const int ptr_1  = ptr[pixel_circle_1];
        const int ptr_9  = ptr[pixel_circle_9];

        if((ptr_1 <= vt && ptr_1 >= v_t && ptr_9 <= vt && ptr_9 >= v_t))
            return;


        n_bright += (ptr_1 > v_t) + (ptr_9 > vt);
        n_dark  += (ptr_1 < v_t) + (ptr_9 < vt);;


        const int pixel_circle_3   =  1*imwidth+3;
        const int pixel_circle_11  = -1*imwidth-3;

        const int ptr_3  = ptr[pixel_circle_3];
        const int ptr_11 = ptr[pixel_circle_11];


        if((ptr_3 <= vt && ptr_3 >= v_t && ptr_11 <= vt && ptr_11 >= v_t))
            return;

        n_bright += (ptr_3 > v_t) + (ptr_11 > vt);
        n_dark  += (ptr_3 < v_t) + (ptr_11 < vt);;




        const int pixel_circle_5   = -1*imwidth+3;
        const int pixel_circle_13 =  1*imwidth-3;

        const int ptr_5  = ptr[pixel_circle_5];
        const int ptr_13 = ptr[pixel_circle_13];

        if((ptr_5 <= vt && ptr_5 >= v_t && ptr_13 <= vt && ptr_13 >= v_t))
            return;

        n_bright += (ptr_5 > v_t) + (ptr_13 > vt);
        n_dark  += (ptr_5 < v_t) + (ptr_13 < vt);;



        const int pixel_circle_7   = -3*imwidth+1;
        const int pixel_circle_15 =  3*imwidth-1;
        const int ptr_7  = ptr[pixel_circle_7];
        const int ptr_15 = ptr[pixel_circle_15];

        if((ptr_7 <= vt && ptr_7 >= v_t && ptr_15 <= vt && ptr_15 >= v_t))
            return;

        n_bright += (ptr_7 > v_t) + (ptr_15 > vt);
        n_dark  += (ptr_7 < v_t) + (ptr_15 < vt);;



        {


            //        int ptr_data[16];

            //        ptr_data[0] = pixel_circle_0;
            //        ptr_data[8] = pixel_circle_8;
            //        ptr_data[4]  = pixel_circle_4;
            //        ptr_data[12]= pixel_circle_12;
            //        ptr_data[2]  = pixel_circle_2;
            //        ptr_data[10] = pixel_circle_10;
            //        ptr_data[6] = pixel_circle_6;
            //        ptr_data[14] = pixel_circle_14;
            //        ptr_data[1]  = pixel_circle_1;
            //        ptr_data[9] = pixel_circle_9;
            //        ptr_data[3] = pixel_circle_3;
            //        ptr_data[11]= pixel_circle_11;
            //        ptr_data[5] = pixel_circle_5;
            //        ptr_data[13]= pixel_circle_13;
            //        ptr_data[7]  = pixel_circle_7;
            //        ptr_data[15] = pixel_circle_15;


            //        ptr_data[0] = ptr[pixel_circle_0];
            //        ptr_data[8] = ptr[pixel_circle_8];
            //        ptr_data[4]  = ptr[pixel_circle_4];
            //        ptr_data[12]= ptr[pixel_circle_12];
            //        ptr_data[2]  = ptr[pixel_circle_2];
            //        ptr_data[10] = ptr[pixel_circle_10];
            //        ptr_data[6] = ptr[pixel_circle_6];
            //        ptr_data[14] = ptr[pixel_circle_14];
            //        ptr_data[1]  = ptr[pixel_circle_1];
            //        ptr_data[9] = ptr[pixel_circle_9];
            //        ptr_data[3] = ptr[pixel_circle_3];
            //        ptr_data[11]= ptr[pixel_circle_11];
            //        ptr_data[5] = ptr[pixel_circle_5];
            //        ptr_data[13]= ptr[pixel_circle_13];
            //        ptr_data[7]  = ptr[pixel_circle_7];
            //        ptr_data[15] = ptr[pixel_circle_15];


            //                const int ptr_0 = pixel_circle_0;
            //                const int ptr_8 = pixel_circle_8;
            //                const int ptr_4  = pixel_circle_4;
            //                const int ptr_12 = pixel_circle_12;
            //                const int ptr_2  = pixel_circle_2;
            //                const int ptr_10 = pixel_circle_10;
            //                const int ptr_6  = pixel_circle_6;
            //                const int ptr_14 = pixel_circle_14;
            //                const int ptr_1  = pixel_circle_1;
            //                const int ptr_9  = pixel_circle_9;
            //                const int ptr_3  = pixel_circle_3;
            //                const int ptr_11 = pixel_circle_11;
            //                const int ptr_5  = pixel_circle_5;
            //                const int ptr_13 = pixel_circle_13;
            //                const int ptr_7  = pixel_circle_7;
            //                const int ptr_15 = pixel_circle_15;


            //        const int ptr_0 = ptr[3*imwidth];
            //        const int ptr_8 = ptr[-3*imwidth];

            //        if(ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t)
            //            return;

            //        const int ptr_4  = ptr[3];
            //        const int ptr_12 = ptr[-3];


            //        if(ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t)
            //            return;

            //        const int imw3 = 3*imwidth;
            //        const int nimw3 = -imw3;
            //        const int imw2 = 2*imwidth;
            //        const int nimw2 = -imw2;
            //        const int nimw = -imwidth;

            //        const int ptr_1  = ptr[imw3+1];
            //        const int ptr_2  = ptr[imw2+2];
            //        const int ptr_3  = ptr[imwidth+3];
            //        const int ptr_5  = ptr[nimw+3];
            //        const int ptr_6  = ptr[nimw2+2];
            //        const int ptr_7  = ptr[nimw3+1];
            //        const int ptr_9  = ptr[nimw3-1];
            //        const int ptr_10 = ptr[nimw2-2];
            //        const int ptr_11 = ptr[nimw-3];
            //        const int ptr_13 = ptr[imwidth-3];
            //        const int ptr_14 = ptr[imw2-2];
            //        const int ptr_15 = ptr[imw3-1];

            //            int n_bright = 0;
            //            int n_dark = 0;

            //            int bright_sum = 0;
            //            int dark_sum = 0;


            //            const int neg_threshold = -threshold;

            {
                //#pragma unroll
                //            for(int i=0;i<16;i++)
                //            {
                //                const int x = ptr_data[i] - v;
                //                const int res1 = x > threshold;
                //                const int res2 = x < neg_threshold;
                //                n_bright += res1;
                //                n_dark += res2;
                //                bright_sum += res1 * x;
                //                dark_sum += res2 * x;

                //                //                n_bright += (x > threshold);
                //                //                n_dark += (x < neg_threshold);
                //                //                bright_sum += (x > threshold) * x;
                //                //                dark_sum += (x < neg_threshold) * x;
                //            }

                //                    {const int x = ptr_0 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_1 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_2 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_3 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_4 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_5 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_6 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_7 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_8 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_9 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_10 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_11 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_12 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_13 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_14 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_15 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    dark_sum *= -1;

                //                {const float x = ptr_0 - v; local_score += fabsf(x);}
                //                {const float x = ptr_1 - v; local_score += fabsf(x);}
                //                {const float x = ptr_2 - v; local_score += fabsf(x);}
                //                {const float x = ptr_3 - v; local_score += fabsf(x);}
                //                {const float x = ptr_4 - v; local_score += fabsf(x);}
                //                {const float x = ptr_5 - v; local_score += fabsf(x);}
                //                {const float x = ptr_6 - v; local_score += fabsf(x);}
                //                {const float x = ptr_7 - v; local_score += fabsf(x);}
                //                {const float x = ptr_8 - v; local_score += fabsf(x);}
                //                {const float x = ptr_9 - v; local_score += fabsf(x);}
                //                {const float x = ptr_10 - v; local_score += fabsf(x);}
                //                {const float x = ptr_11 - v; local_score += fabsf(x);}
                //                {const float x = ptr_12 - v; local_score += fabsf(x);}
                //                {const float x = ptr_13 - v; local_score += fabsf(x);}
                //                {const float x = ptr_14 - v; local_score += fabsf(x);}
                //                {const float x = ptr_15 - v; local_score += fabsf(x);}

                float sum  = fabsf(ptr_0 - v) + fabsf(ptr_1 - v)
                        + fabsf(ptr_2 - v) + fabsf(ptr_3 - v)
                        + fabsf(ptr_4 - v) + fabsf(ptr_5 - v)
                        + fabsf(ptr_6 - v) + fabsf(ptr_7 - v)
                        + fabsf(ptr_8 - v) + fabsf(ptr_9 - v)
                        + fabsf(ptr_10 - v) + fabsf(ptr_11 - v)
                        + fabsf(ptr_12 - v) + fabsf(ptr_13 - v)
                        + fabsf(ptr_14 - v) + fabsf(ptr_15 - v);



                //            {if(ptr_0 > vt){n_bright++;bright_sum += ptr_0-v;} else if(ptr_0 < v_t){n_dark++; dark_sum += ptr_0-v;}}


                //                        {if(ptr_0 > vt){n_bright++;bright_sum += ptr_0-v;} else if(ptr_0 < v_t){n_dark++; dark_sum += ptr_0-v;}}
                //                        {if(ptr_1 > vt){n_bright++;bright_sum += ptr_1-v;} else if(ptr_1 < v_t){n_dark++; dark_sum += ptr_1-v;}}
                //                        {if(ptr_2 > vt){n_bright++;bright_sum += ptr_2-v;} else if(ptr_2 < v_t){n_dark++; dark_sum += ptr_2-v;}}
                //                        {if(ptr_3 > vt){n_bright++;bright_sum += ptr_3-v;} else if(ptr_3 < v_t){n_dark++; dark_sum += ptr_3-v;}}
                //                        {if(ptr_4 > vt){n_bright++;bright_sum += ptr_4-v;} else if(ptr_4 < v_t){n_dark++; dark_sum += ptr_4-v;}}
                //                        {if(ptr_5 > vt){n_bright++;bright_sum += ptr_5-v;} else if(ptr_5 < v_t){n_dark++; dark_sum += ptr_5-v;}}
                //                        {if(ptr_6 > vt){n_bright++;bright_sum += ptr_6-v;} else if(ptr_6 < v_t){n_dark++; dark_sum += ptr_6-v;}}
                //                        {if(ptr_7 > vt){n_bright++;bright_sum += ptr_7-v;} else if(ptr_7 < v_t){n_dark++; dark_sum += ptr_7-v;}}
                //                        {if(ptr_8 > vt){n_bright++;bright_sum += ptr_8-v;} else if(ptr_8 < v_t){n_dark++; dark_sum += ptr_8-v;}}
                //                        {if(ptr_9 > vt){n_bright++;bright_sum += ptr_9-v;} else if(ptr_9 < v_t){n_dark++; dark_sum += ptr_9;}}
                //                        {if(ptr_10 > vt){n_bright++;bright_sum += ptr_10-v;} else if(ptr_10 < v_t){n_dark++; dark_sum += ptr_10-v;}}
                //                        {if(ptr_11 > vt){n_bright++;bright_sum += ptr_11-v;} else if(ptr_11 < v_t){n_dark++; dark_sum += ptr_11-v;}}
                //                        {if(ptr_12 > vt){n_bright++;bright_sum += ptr_12-v;} else if(ptr_12 < v_t){n_dark++; dark_sum += ptr_12-v;}}
                //                        {if(ptr_13 > vt){n_bright++;bright_sum += ptr_13-v;} else if(ptr_13 < v_t){n_dark++; dark_sum += ptr_13-v;}}
                //                        {if(ptr_14 > vt){n_bright++;bright_sum += ptr_14-v;} else if(ptr_14 < v_t){n_dark++; dark_sum += ptr_14-v;}}
                //                        {if(ptr_15 > vt){n_bright++;bright_sum += ptr_15-v;} else if(ptr_15 < v_t){n_dark++; dark_sum += ptr_15-v;}}

                //                   unsigned int bright_list = 0;
                //                   unsigned int dark_list = 0;

                //                   bright_list |= (ptr_8 > vt); bright_list << 1;
                //                   bright_list |= (ptr_9 > vt); bright_list << 1;
                //                   bright_list |= (ptr_10 > vt); bright_list << 1;
                //                   bright_list |= (ptr_11 > vt); bright_list << 1;
                //                   bright_list |= (ptr_12 > vt); bright_list << 1;
                //                   bright_list |= (ptr_13 > vt); bright_list << 1;
                //                   bright_list |= (ptr_14 > vt); bright_list << 1;
                //                   bright_list |= (ptr_15 > vt); bright_list << 1;
                //                   bright_list |= (ptr_0 > vt); bright_list << 1;
                //                   bright_list |= (ptr_1 > vt); bright_list << 1;
                //                   bright_list |= (ptr_2 > vt); bright_list << 1;
                //                   bright_list |= (ptr_3 > vt); bright_list << 1;
                //                   bright_list |= (ptr_4 > vt); bright_list << 1;
                //                   bright_list |= (ptr_5 > vt); bright_list << 1;
                //                   bright_list |= (ptr_6 > vt); bright_list << 1;
                //                   bright_list |= (ptr_7 > vt);


                //                   dark_list |= (ptr_8 < v_t);   dark_list << 1;
                //                   dark_list |= (ptr_9 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_10 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_11 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_12 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_13 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_14 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_15 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_0 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_1 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_2 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_3 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_4 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_5 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_6 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_7 < v_t);

                //                    int n_bright = __popc(bright_list);
                //                    int n_dark = __popc(dark_list);


                ////divergence less
                //// better because if FAST_N_MIN is less than 0.50, n_bright and n_dark can be same;
                //   const int in_range_bright = (n_bright > n_dark) && (n_bright >= th_FAST_MIN) && (n_bright <= th_FAST_MAX);
                //   const int in_range_dark =  (n_bright < n_dark) &&  (n_dark >= th_FAST_MIN) && (n_dark <= th_FAST_MAX);
                //// this sum of absolute differences, already there in the literature
                //  local_score = in_range_bright * bright_sum + in_range_dark * dark_sum;

                if((n_bright > n_dark) && (n_bright >= th_FAST_MIN) && (n_bright <= th_FAST_MAX))
                    local_score = sum;
                else if((n_bright < n_dark) &&  (n_dark >= th_FAST_MIN) && (n_dark <= th_FAST_MAX))
                    local_score = sum;

                //// was thinking to scale the scores. But it's not a good idea!!
                //  local_score = in_range_bright * n_bright * bright_sum + in_range_dark * n_dark * dark_sum;
            }

            score_data[offset] =  local_score;

        }
    }

}
//        __syncthreads();
}




__global__ void FASTComputeScoreGPU_patternSize_16(int n_threads,
                                                   int imheight, int imwidth,
                                                   int roiheight, int roiwidth,
                                                   int threshold,
                                                   const unsigned char* __restrict__ image_data,
                                                   int* __restrict__ score_data)
{

#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if(w < imwidth && h < imheight)
    {
        const int h_im = h;// + BORDER_SKIP;
        const int w_im = w;// + BORDER_SKIP;

        const int offset = h_im * imwidth + w_im;

        score_data[offset] =  0;

        if(w >= BORDER_SKIP && w < imwidth - BORDER_SKIP && h < imheight - BORDER_SKIP && h >= BORDER_SKIP)
        {

#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int h = index / roiwidth;
        int w = index % roiwidth;
#endif


        float local_score = 0;

        const unsigned char* ptr = image_data + offset;

        const int v = ptr[0];

        const int vt = v + threshold;
        const int v_t = v - threshold;

        int n_bright = 0;
        int n_dark = 0;

        const int pixel_circle_0  =  3*imwidth+0;
        const int pixel_circle_8  = -3*imwidth+0;

        const int ptr_0 = ptr[pixel_circle_0];
        const int ptr_8 = ptr[pixel_circle_8];

        if((ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t))
            return;

        const int pixel_circle_4  =  0*imwidth+3;
        const int pixel_circle_12 =  0*imwidth-3;
        const int ptr_4  = ptr[pixel_circle_4];
        const int ptr_12 = ptr[pixel_circle_12];


        if((ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t))
            return;


        const int pixel_circle_6  = -2*imwidth+2;
        const int pixel_circle_14 =  2*imwidth-2;

        const int ptr_6  = ptr[pixel_circle_6];
        const int ptr_14 = ptr[pixel_circle_14];


        const int pixel_circle_2  =  2*imwidth+2;
        const int pixel_circle_10 = -2*imwidth-2;
        const int ptr_2  = ptr[pixel_circle_2];
        const int ptr_10 = ptr[pixel_circle_10];


        const int pixel_circle_1   =  3*imwidth+1;
        const int pixel_circle_9   = -3*imwidth-1;

        const int ptr_1  = ptr[pixel_circle_1];
        const int ptr_9  = ptr[pixel_circle_9];

        const int pixel_circle_3   =  1*imwidth+3;
        const int pixel_circle_11  = -1*imwidth-3;

        const int ptr_3  = ptr[pixel_circle_3];
        const int ptr_11 = ptr[pixel_circle_11];

        const int pixel_circle_5   = -1*imwidth+3;
        const int pixel_circle_13 =  1*imwidth-3;

        const int ptr_5  = ptr[pixel_circle_5];
        const int ptr_13 = ptr[pixel_circle_13];

        const int pixel_circle_7   = -3*imwidth+1;
        const int pixel_circle_15 =  3*imwidth-1;
        const int ptr_7  = ptr[pixel_circle_7];
        const int ptr_15 = ptr[pixel_circle_15];

        {


            {

                int n_bright = 0;
                int n_dark = 0;

                int bright_sum = 0;
                int dark_sum = 0;


                const int neg_threshold = -threshold;

                int prev = 0;

                {
                    int x = ptr_0 - v;
                    if(x > threshold){ prev = 0;n_bright++; bright_sum += x;}
                    else if(x < neg_threshold){prev = 1;n_dark++; dark_sum += x;}
                }

                int x = ptr_1 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_2 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_3 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_4 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_5 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_6 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_7 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_8 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_9 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_10 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_11 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_12 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_13 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_14 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}
                x = ptr_15 - v;if(x > threshold){n_bright++;bright_sum += x;if(prev == 1){n_dark = 0;dark_sum = 0;prev=0;}}else if(x < neg_threshold){n_dark++;dark_sum += x;if(prev == 0){n_bright = 0;bright_sum = 0;prev=1;}}



                dark_sum *= -1;

                if((n_bright > n_dark) && (n_bright >= th_FAST_MIN) && (n_bright <= th_FAST_MAX))
                    local_score = bright_sum;
                else if((n_bright < n_dark) &&  (n_dark >= th_FAST_MIN) && (n_dark <= th_FAST_MAX))
                    local_score = dark_sum;
            }


            //        int ptr_data[16];

            //        ptr_data[0] = pixel_circle_0;
            //        ptr_data[8] = pixel_circle_8;
            //        ptr_data[4]  = pixel_circle_4;
            //        ptr_data[12]= pixel_circle_12;
            //        ptr_data[2]  = pixel_circle_2;
            //        ptr_data[10] = pixel_circle_10;
            //        ptr_data[6] = pixel_circle_6;
            //        ptr_data[14] = pixel_circle_14;
            //        ptr_data[1]  = pixel_circle_1;
            //        ptr_data[9] = pixel_circle_9;
            //        ptr_data[3] = pixel_circle_3;
            //        ptr_data[11]= pixel_circle_11;
            //        ptr_data[5] = pixel_circle_5;
            //        ptr_data[13]= pixel_circle_13;
            //        ptr_data[7]  = pixel_circle_7;
            //        ptr_data[15] = pixel_circle_15;


            //        ptr_data[0] = ptr[pixel_circle_0];
            //        ptr_data[8] = ptr[pixel_circle_8];
            //        ptr_data[4]  = ptr[pixel_circle_4];
            //        ptr_data[12]= ptr[pixel_circle_12];
            //        ptr_data[2]  = ptr[pixel_circle_2];
            //        ptr_data[10] = ptr[pixel_circle_10];
            //        ptr_data[6] = ptr[pixel_circle_6];
            //        ptr_data[14] = ptr[pixel_circle_14];
            //        ptr_data[1]  = ptr[pixel_circle_1];
            //        ptr_data[9] = ptr[pixel_circle_9];
            //        ptr_data[3] = ptr[pixel_circle_3];
            //        ptr_data[11]= ptr[pixel_circle_11];
            //        ptr_data[5] = ptr[pixel_circle_5];
            //        ptr_data[13]= ptr[pixel_circle_13];
            //        ptr_data[7]  = ptr[pixel_circle_7];
            //        ptr_data[15] = ptr[pixel_circle_15];


            //                const int ptr_0 = pixel_circle_0;
            //                const int ptr_8 = pixel_circle_8;
            //                const int ptr_4  = pixel_circle_4;
            //                const int ptr_12 = pixel_circle_12;
            //                const int ptr_2  = pixel_circle_2;
            //                const int ptr_10 = pixel_circle_10;
            //                const int ptr_6  = pixel_circle_6;
            //                const int ptr_14 = pixel_circle_14;
            //                const int ptr_1  = pixel_circle_1;
            //                const int ptr_9  = pixel_circle_9;
            //                const int ptr_3  = pixel_circle_3;
            //                const int ptr_11 = pixel_circle_11;
            //                const int ptr_5  = pixel_circle_5;
            //                const int ptr_13 = pixel_circle_13;
            //                const int ptr_7  = pixel_circle_7;
            //                const int ptr_15 = pixel_circle_15;


            //        const int ptr_0 = ptr[3*imwidth];
            //        const int ptr_8 = ptr[-3*imwidth];

            //        if(ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t)
            //            return;

            //        const int ptr_4  = ptr[3];
            //        const int ptr_12 = ptr[-3];


            //        if(ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t)
            //            return;

            //        const int imw3 = 3*imwidth;
            //        const int nimw3 = -imw3;
            //        const int imw2 = 2*imwidth;
            //        const int nimw2 = -imw2;
            //        const int nimw = -imwidth;

            //        const int ptr_1  = ptr[imw3+1];
            //        const int ptr_2  = ptr[imw2+2];
            //        const int ptr_3  = ptr[imwidth+3];
            //        const int ptr_5  = ptr[nimw+3];
            //        const int ptr_6  = ptr[nimw2+2];
            //        const int ptr_7  = ptr[nimw3+1];
            //        const int ptr_9  = ptr[nimw3-1];
            //        const int ptr_10 = ptr[nimw2-2];
            //        const int ptr_11 = ptr[nimw-3];
            //        const int ptr_13 = ptr[imwidth-3];
            //        const int ptr_14 = ptr[imw2-2];
            //        const int ptr_15 = ptr[imw3-1];

            //            int n_bright = 0;
            //            int n_dark = 0;

            //            int bright_sum = 0;
            //            int dark_sum = 0;


            //            const int neg_threshold = -threshold;

            {
                //#pragma unroll
                //            for(int i=0;i<16;i++)
                //            {
                //                const int x = ptr_data[i] - v;
                //                const int res1 = x > threshold;
                //                const int res2 = x < neg_threshold;
                //                n_bright += res1;
                //                n_dark += res2;
                //                bright_sum += res1 * x;
                //                dark_sum += res2 * x;

                //                //                n_bright += (x > threshold);
                //                //                n_dark += (x < neg_threshold);
                //                //                bright_sum += (x > threshold) * x;
                //                //                dark_sum += (x < neg_threshold) * x;
                //            }

                //                    {const int x = ptr_0 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_1 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_2 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_3 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_4 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_5 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_6 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_7 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_8 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_9 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_10 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_11 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_12 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_13 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_14 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    {const int x = ptr_15 - v; const int res1 = x > threshold; const int res2 = x < neg_threshold; n_bright += res1; n_dark += res2; bright_sum += res1 * x; dark_sum += res2 * x;}
                //                    dark_sum *= -1;

                //                {const float x = ptr_0 - v; local_score += fabsf(x);}
                //                {const float x = ptr_1 - v; local_score += fabsf(x);}
                //                {const float x = ptr_2 - v; local_score += fabsf(x);}
                //                {const float x = ptr_3 - v; local_score += fabsf(x);}
                //                {const float x = ptr_4 - v; local_score += fabsf(x);}
                //                {const float x = ptr_5 - v; local_score += fabsf(x);}
                //                {const float x = ptr_6 - v; local_score += fabsf(x);}
                //                {const float x = ptr_7 - v; local_score += fabsf(x);}
                //                {const float x = ptr_8 - v; local_score += fabsf(x);}
                //                {const float x = ptr_9 - v; local_score += fabsf(x);}
                //                {const float x = ptr_10 - v; local_score += fabsf(x);}
                //                {const float x = ptr_11 - v; local_score += fabsf(x);}
                //                {const float x = ptr_12 - v; local_score += fabsf(x);}
                //                {const float x = ptr_13 - v; local_score += fabsf(x);}
                //                {const float x = ptr_14 - v; local_score += fabsf(x);}
                //                {const float x = ptr_15 - v; local_score += fabsf(x);}

                //                float sum  = fabsf(ptr_0 - v) + fabsf(ptr_1 - v)
                //                        + fabsf(ptr_2 - v) + fabsf(ptr_3 - v)
                //                        + fabsf(ptr_4 - v) + fabsf(ptr_5 - v)
                //                        + fabsf(ptr_6 - v) + fabsf(ptr_7 - v)
                //                        + fabsf(ptr_8 - v) + fabsf(ptr_9 - v)
                //                        + fabsf(ptr_10 - v) + fabsf(ptr_11 - v)
                //                        + fabsf(ptr_12 - v) + fabsf(ptr_13 - v)
                //                        + fabsf(ptr_14 - v) + fabsf(ptr_15 - v);



                //            {if(ptr_0 > vt){n_bright++;bright_sum += ptr_0-v;} else if(ptr_0 < v_t){n_dark++; dark_sum += ptr_0-v;}}


                //                        {if(ptr_0 > vt){n_bright++;bright_sum += ptr_0-v;} else if(ptr_0 < v_t){n_dark++; dark_sum += ptr_0-v;}}
                //                        {if(ptr_1 > vt){n_bright++;bright_sum += ptr_1-v;} else if(ptr_1 < v_t){n_dark++; dark_sum += ptr_1-v;}}
                //                        {if(ptr_2 > vt){n_bright++;bright_sum += ptr_2-v;} else if(ptr_2 < v_t){n_dark++; dark_sum += ptr_2-v;}}
                //                        {if(ptr_3 > vt){n_bright++;bright_sum += ptr_3-v;} else if(ptr_3 < v_t){n_dark++; dark_sum += ptr_3-v;}}
                //                        {if(ptr_4 > vt){n_bright++;bright_sum += ptr_4-v;} else if(ptr_4 < v_t){n_dark++; dark_sum += ptr_4-v;}}
                //                        {if(ptr_5 > vt){n_bright++;bright_sum += ptr_5-v;} else if(ptr_5 < v_t){n_dark++; dark_sum += ptr_5-v;}}
                //                        {if(ptr_6 > vt){n_bright++;bright_sum += ptr_6-v;} else if(ptr_6 < v_t){n_dark++; dark_sum += ptr_6-v;}}
                //                        {if(ptr_7 > vt){n_bright++;bright_sum += ptr_7-v;} else if(ptr_7 < v_t){n_dark++; dark_sum += ptr_7-v;}}
                //                        {if(ptr_8 > vt){n_bright++;bright_sum += ptr_8-v;} else if(ptr_8 < v_t){n_dark++; dark_sum += ptr_8-v;}}
                //                        {if(ptr_9 > vt){n_bright++;bright_sum += ptr_9-v;} else if(ptr_9 < v_t){n_dark++; dark_sum += ptr_9;}}
                //                        {if(ptr_10 > vt){n_bright++;bright_sum += ptr_10-v;} else if(ptr_10 < v_t){n_dark++; dark_sum += ptr_10-v;}}
                //                        {if(ptr_11 > vt){n_bright++;bright_sum += ptr_11-v;} else if(ptr_11 < v_t){n_dark++; dark_sum += ptr_11-v;}}
                //                        {if(ptr_12 > vt){n_bright++;bright_sum += ptr_12-v;} else if(ptr_12 < v_t){n_dark++; dark_sum += ptr_12-v;}}
                //                        {if(ptr_13 > vt){n_bright++;bright_sum += ptr_13-v;} else if(ptr_13 < v_t){n_dark++; dark_sum += ptr_13-v;}}
                //                        {if(ptr_14 > vt){n_bright++;bright_sum += ptr_14-v;} else if(ptr_14 < v_t){n_dark++; dark_sum += ptr_14-v;}}
                //                        {if(ptr_15 > vt){n_bright++;bright_sum += ptr_15-v;} else if(ptr_15 < v_t){n_dark++; dark_sum += ptr_15-v;}}

                //                   unsigned int bright_list = 0;
                //                   unsigned int dark_list = 0;

                //                   bright_list |= (ptr_8 > vt); bright_list << 1;
                //                   bright_list |= (ptr_9 > vt); bright_list << 1;
                //                   bright_list |= (ptr_10 > vt); bright_list << 1;
                //                   bright_list |= (ptr_11 > vt); bright_list << 1;
                //                   bright_list |= (ptr_12 > vt); bright_list << 1;
                //                   bright_list |= (ptr_13 > vt); bright_list << 1;
                //                   bright_list |= (ptr_14 > vt); bright_list << 1;
                //                   bright_list |= (ptr_15 > vt); bright_list << 1;
                //                   bright_list |= (ptr_0 > vt); bright_list << 1;
                //                   bright_list |= (ptr_1 > vt); bright_list << 1;
                //                   bright_list |= (ptr_2 > vt); bright_list << 1;
                //                   bright_list |= (ptr_3 > vt); bright_list << 1;
                //                   bright_list |= (ptr_4 > vt); bright_list << 1;
                //                   bright_list |= (ptr_5 > vt); bright_list << 1;
                //                   bright_list |= (ptr_6 > vt); bright_list << 1;
                //                   bright_list |= (ptr_7 > vt);


                //                   dark_list |= (ptr_8 < v_t);   dark_list << 1;
                //                   dark_list |= (ptr_9 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_10 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_11 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_12 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_13 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_14 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_15 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_0 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_1 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_2 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_3 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_4 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_5 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_6 < v_t);  dark_list << 1;
                //                   dark_list |= (ptr_7 < v_t);

                //                    int n_bright = __popc(bright_list);
                //                    int n_dark = __popc(dark_list);


                ////divergence less
                //// better because if FAST_N_MIN is less than 0.50, n_bright and n_dark can be same;
                //   const int in_range_bright = (n_bright > n_dark) && (n_bright >= th_FAST_MIN) && (n_bright <= th_FAST_MAX);
                //   const int in_range_dark =  (n_bright < n_dark) &&  (n_dark >= th_FAST_MIN) && (n_dark <= th_FAST_MAX);
                //// this sum of absolute differences, already there in the literature
                //  local_score = in_range_bright * bright_sum + in_range_dark * dark_sum;

                //                if((n_bright > n_dark) && (n_bright >= th_FAST_MIN) && (n_bright <= th_FAST_MAX))
                //                    local_score = sum;
                //                else if((n_bright < n_dark) &&  (n_dark >= th_FAST_MIN) && (n_dark <= th_FAST_MAX))
                //                    local_score = sum;

                //// was thinking to scale the scores. But it's not a good idea!!
                //  local_score = in_range_bright * n_bright * bright_sum + in_range_dark * n_dark * dark_sum;
            }

            score_data[offset] =  local_score;

        }
    }

}
//        __syncthreads();
}



void ORB_GPU::FAST_compute_score(int height, int width,
                                 unsigned char* image_data_gpu,
                                 int* threshold_tab,
                                 int* score_data_gpu, cudaStream_t& cuda_stream)
{

    //    int roiheight = (height - 2*BORDER_SKIP);
    //    int roiwidth  = (width - 2*BORDER_SKIP);

    int roiheight = (height - 2*BORDER_SKIP);
    int roiwidth  = (width - 2*BORDER_SKIP);

    int n_threads =  roiheight * roiwidth;


#ifdef GRID_LAUNCH

    {
        int CUDA_NUM_BLOCKS_x = (roiwidth  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
        int CUDA_NUM_BLOCKS_y = (roiheight - 1) / CUDA_NUM_THREADS_PER_BLOCK_y + 1;

        dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
        dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, CUDA_NUM_THREADS_PER_BLOCK_y, 1);


        FASTComputeScoreGPU_patternSize_16<<<grid_dim, block_dim, 0, cuda_stream>>>(
                                                                                      n_threads,
                                                                                      height, width,
                                                                                      roiheight, roiwidth,
                                                                                      threshold_,
                                                                                      image_data_gpu,
                                                                                      score_data_gpu);
    }

#else
    // slower 10-12us
    {

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1 ;

        FASTComputeScoreGPU_patternSize_16<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                              n_threads,
                                                                                                              height, width,
                                                                                                              roiheight, roiwidth,
                                                                                                              threshold_,
                                                                                                              image_data_gpu,
                                                                                                              score_data_gpu);

    }

#endif

}







__global__ void FASTComputeScoreGPU_patternSize_16_lookup(int imheight, int imwidth,
                                                          int threshold,
                                                          int* lookup_table,
                                                          const unsigned char* image_data,
                                                          int image_pitch,
                                                          int* score_data,
                                                          int score_pitch)
{

#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;


    if(w < imwidth && h < imheight)
    {
        const int h_im = h;// + BORDER_SKIP;
        const int w_im = w;// + BORDER_SKIP;

        const int offset = h_im * image_pitch + w_im;
        const int score_offset = h_im * score_pitch + w_im;

#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int h = index / roiwidth;
        int w = index % roiwidth;
#endif

        if(w >= BORDER_SKIP && w < imwidth - BORDER_SKIP && h < imheight - BORDER_SKIP && h >= BORDER_SKIP)
        {

            int local_score = 0;

            const unsigned char* ptr = image_data + offset;

            const int v = ptr[0];

            const int vt = v + threshold;
            const int v_t = v - threshold;

            //        const int pixel_circle_0  =  3*imwidth+0;
            //        const int pixel_circle_8  = -3*imwidth+0;

            //        const int ptr_0 = ptr[pixel_circle_0];
            //        const int ptr_8 = ptr[pixel_circle_8];

            //        if((ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t))
            //            return;

            //        const int pixel_circle_4  =  0*imwidth+3;
            //        const int pixel_circle_12 =  0*imwidth-3;
            //        const int ptr_4  = ptr[pixel_circle_4];
            //        const int ptr_12 = ptr[pixel_circle_12];


            //        if((ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t))
            //            return;


            //        const int pixel_circle_6  = -2*imwidth+2;
            //        const int pixel_circle_14 =  2*imwidth-2;

            //        const int ptr_6  = ptr[pixel_circle_6];
            //        const int ptr_14 = ptr[pixel_circle_14];


            //        const int pixel_circle_2  =  2*imwidth+2;
            //        const int pixel_circle_10 = -2*imwidth-2;
            //        const int ptr_2  = ptr[pixel_circle_2];
            //        const int ptr_10 = ptr[pixel_circle_10];


            //        const int pixel_circle_1   =  3*imwidth+1;
            //        const int pixel_circle_9   = -3*imwidth-1;

            //        const int ptr_1  = ptr[pixel_circle_1];
            //        const int ptr_9  = ptr[pixel_circle_9];

            //        const int pixel_circle_3   =  1*imwidth+3;
            //        const int pixel_circle_11  = -1*imwidth-3;

            //        const int ptr_3  = ptr[pixel_circle_3];
            //        const int ptr_11 = ptr[pixel_circle_11];

            //        const int pixel_circle_5   = -1*imwidth+3;
            //        const int pixel_circle_13 =  1*imwidth-3;

            //        const int ptr_5  = ptr[pixel_circle_5];
            //        const int ptr_13 = ptr[pixel_circle_13];

            //        const int pixel_circle_7   = -3*imwidth+1;
            //        const int pixel_circle_15 =  3*imwidth-1;
            //        const int ptr_7  = ptr[pixel_circle_7];
            //        const int ptr_15 = ptr[pixel_circle_15];


            //                const int imw3 = 3*imwidth;
            //                const int imw2 = 2*imwidth;

            //                const int pixel_circle_4  =  3;
            //                const int pixel_circle_12 = -3;
            //                const int ptr_4  = ptr[pixel_circle_4];
            //                const int ptr_12 = ptr[pixel_circle_12];


            //                if((ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t))
            //                    return;


            //                const int pixel_circle_0  =  imw3;
            //                const int pixel_circle_8  = -imw3;

            //                const int ptr_0 = ptr[pixel_circle_0];
            //                const int ptr_8 = ptr[pixel_circle_8];

            //                if((ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t))
            //                    return;


            //                const int pixel_circle_6  = -imw2+2;
            //                const int pixel_circle_14 =  imw2-2;

            //                const int ptr_6  = ptr[pixel_circle_6];
            //                const int ptr_14 = ptr[pixel_circle_14];


            //                const int pixel_circle_2  =  imw2+2;
            //                const int pixel_circle_10 = -imw2-2;
            //                const int ptr_2  = ptr[pixel_circle_2];
            //                const int ptr_10 = ptr[pixel_circle_10];


            //                const int pixel_circle_1   =  imw3+1;
            //                const int pixel_circle_9   = -imw3-1;

            //                const int ptr_1  = ptr[pixel_circle_1];
            //                const int ptr_9  = ptr[pixel_circle_9];

            //                const int pixel_circle_3   =  imwidth+3;
            //                const int pixel_circle_11  = -imwidth-3;

            //                const int ptr_3  = ptr[pixel_circle_3];
            //                const int ptr_11 = ptr[pixel_circle_11];

            //                const int pixel_circle_5   = -imwidth+3;
            //                const int pixel_circle_13 =  imwidth-3;

            //                const int ptr_5  = ptr[pixel_circle_5];
            //                const int ptr_13 = ptr[pixel_circle_13];

            //                const int pixel_circle_7   = -imw3+1;
            //                const int pixel_circle_15 =   imw3-1;
            //                const int ptr_7  = ptr[pixel_circle_7];
            //                const int ptr_15 = ptr[pixel_circle_15];



            const int ptr_4  = ptr[3];
            const int ptr_12 = ptr[-3];


            if(ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t)
            {        score_data[score_offset] =  0;  return;}


            const int imw3 = 3*image_pitch;
            const int nimw3 = -imw3;

            const int ptr_0 = ptr[imw3];
            const int ptr_8 = ptr[nimw3];

            if(ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t)
            {        score_data[score_offset] =  0;return;}


            const int imw2 = 2*image_pitch;
            const int nimw2 = -imw2;
            const int nimw = -image_pitch;

            const int ptr_1  = ptr[imw3+1];
            const int ptr_2  = ptr[imw2+2];
            const int ptr_3  = ptr[image_pitch+3];
            const int ptr_5  = ptr[nimw+3];
            const int ptr_6  = ptr[nimw2+2];
            const int ptr_7  = ptr[nimw3+1];
            const int ptr_9  = ptr[nimw3-1];
            const int ptr_10 = ptr[nimw2-2];
            const int ptr_11 = ptr[nimw-3];
            const int ptr_13 = ptr[image_pitch-3];
            const int ptr_14 = ptr[imw2-2];
            const int ptr_15 = ptr[imw3-1];

            {
                {
                    int bright_idx = 0;
                    int dark_idx = 0;

                    {
                        //                                        bright_idx = (ptr_0 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_1 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_2 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_3 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_4 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_5 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_6 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_7 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_8 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_9 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_10 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_11 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_12 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_13 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_14 > vt); bright_idx <<= 1;
                        //                                        bright_idx |= (ptr_15 > vt);

                        //                                        dark_idx = (ptr_0 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_1 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_2 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_3 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_4 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_5 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_6 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_7 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_8 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_9 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_10 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_11 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_12 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_13 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_14 < v_t); dark_idx <<= 1;
                        //                                        dark_idx |= (ptr_15 < v_t);

                        //                                        bright_idx = (ptr_0 > vt);
                        //                                        bright_idx |= (ptr_1 > vt) << 1;
                        //                                        bright_idx |= (ptr_2 > vt) << 2;
                        //                                        bright_idx |= (ptr_3 > vt) << 3;
                        //                                        bright_idx |= (ptr_4 > vt) << 4;
                        //                                        bright_idx |= (ptr_5 > vt) << 5;
                        //                                        bright_idx |= (ptr_6 > vt) << 6;;
                        //                                        bright_idx |= (ptr_7 > vt) << 7;
                        //                                        bright_idx |= (ptr_8 > vt) << 8;
                        //                                        bright_idx |= (ptr_9 > vt) << 9;
                        //                                        bright_idx |= (ptr_10 > vt) << 10;
                        //                                        bright_idx |= (ptr_11 > vt) << 11;
                        //                                        bright_idx |= (ptr_12 > vt) << 12;
                        //                                        bright_idx |= (ptr_13 > vt) << 13;
                        //                                        bright_idx |= (ptr_14 > vt) << 14;
                        //                                        bright_idx |= (ptr_15 > vt) << 15;

                        //                                        dark_idx = (ptr_0 < v_t);
                        //                                        dark_idx |= (ptr_1 < v_t) << 1;
                        //                                        dark_idx |= (ptr_2 < v_t) << 2;
                        //                                        dark_idx |= (ptr_3 < v_t) << 3;
                        //                                        dark_idx |= (ptr_4 < v_t) << 4;
                        //                                        dark_idx |= (ptr_5 < v_t) << 5;
                        //                                        dark_idx |= (ptr_6 < v_t) << 6;
                        //                                        dark_idx |= (ptr_7 < v_t) << 7;
                        //                                        dark_idx |= (ptr_8 < v_t) << 8;
                        //                                        dark_idx |= (ptr_9 < v_t) << 9;
                        //                                        dark_idx |= (ptr_10 < v_t) << 10;
                        //                                        dark_idx |= (ptr_11 < v_t) << 11;
                        //                                        dark_idx |= (ptr_12 < v_t) << 12;
                        //                                        dark_idx |= (ptr_13 < v_t) << 13;
                        //                                        dark_idx |= (ptr_14 < v_t) << 14;
                        //                                        dark_idx |= (ptr_15 < v_t) << 15;


                        //                                                            bright_idx = (ptr_0 > vt)
                        //                                                            | ((ptr_1 > vt) << 1)
                        //                                                            | ((ptr_2 > vt) << 2)
                        //                                                            | ((ptr_3 > vt) << 3)
                        //                                                            | ((ptr_4 > vt) << 4)
                        //                                                            | ((ptr_5 > vt) << 5)
                        //                                                            | ((ptr_6 > vt) << 6)
                        //                                                            | ((ptr_7 > vt) << 7)
                        //                                                            | ((ptr_8 > vt) << 8)
                        //                                                            | ((ptr_9 > vt) << 9)
                        //                                                            | ((ptr_10 > vt) << 10)
                        //                                                            | ((ptr_11 > vt) << 11)
                        //                                                            | ((ptr_12 > vt) << 12)
                        //                                                            | ((ptr_13 > vt) << 13)
                        //                                                            | ((ptr_14 > vt) << 14)
                        //                                                            | ((ptr_15 > vt) << 15);

                        //                                                            dark_idx = (ptr_0 < v_t)
                        //                                                            | ((ptr_1 < v_t) << 1)
                        //                                                            | ((ptr_2 < v_t) << 2)
                        //                                                            | ((ptr_3 < v_t) << 3)
                        //                                                            | ((ptr_4 < v_t) << 4)
                        //                                                            | ((ptr_5 < v_t) << 5)
                        //                                                            | ((ptr_6 < v_t) << 6)
                        //                                                            | ((ptr_7 < v_t) << 7)
                        //                                                            | ((ptr_8 < v_t) << 8)
                        //                                                            | ((ptr_9 < v_t) << 9)
                        //                                                            | ((ptr_10 < v_t) << 10)
                        //                                                            | ((ptr_11 < v_t) << 11)
                        //                                                            | ((ptr_12 < v_t) << 12)
                        //                                                            | ((ptr_13 < v_t) << 13)
                        //                                                            | ((ptr_14 < v_t) << 14)
                        //                                                            | ((ptr_15 < v_t) << 15);





                        //                    bright_idx = (ptr_0 > vt);
                        //                    bright_idx |= (ptr_1 > vt)  ? 0x00000002 : 0x00000000;
                        //                    bright_idx |= (ptr_2 > vt)  ? 0x00000004 : 0x00000000;
                        //                    bright_idx |= (ptr_3 > vt)  ? 0x00000008 : 0x00000000;
                        //                    bright_idx |= (ptr_4 > vt)  ? 0x00000010 : 0x00000000;
                        //                    bright_idx |= (ptr_5 > vt)  ? 0x00000020 : 0x00000000;
                        //                    bright_idx |= (ptr_6 > vt)  ? 0x00000040 : 0x00000000;
                        //                    bright_idx |= (ptr_7 > vt)  ? 0x00000080 : 0x00000000;
                        //                    bright_idx |= (ptr_8 > vt)  ? 0x00000100 : 0x00000000;
                        //                    bright_idx |= (ptr_9 > vt)  ? 0x00000200 : 0x00000000;
                        //                    bright_idx |= (ptr_10 > vt) ? 0x00000400 : 0x00000000;
                        //                    bright_idx |= (ptr_11 > vt) ? 0x00000800 : 0x00000000;
                        //                    bright_idx |= (ptr_12 > vt) ? 0x00001000 : 0x00000000;
                        //                    bright_idx |= (ptr_13 > vt) ? 0x00002000 : 0x00000000;
                        //                    bright_idx |= (ptr_14 > vt) ? 0x00004000 : 0x00000000;
                        //                    bright_idx |= (ptr_15 > vt) ? 0x00008000 : 0x00000000;

                        //                    dark_idx = (ptr_0 < v_t);
                        //                    dark_idx |= (ptr_1 < v_t)  ? 0x00000002 : 0x00000000;
                        //                    dark_idx |= (ptr_2 < v_t)  ? 0x00000004 : 0x00000000;
                        //                    dark_idx |= (ptr_3 < v_t)  ? 0x00000008 : 0x00000000;
                        //                    dark_idx |= (ptr_4 < v_t)  ? 0x00000010 : 0x00000000;
                        //                    dark_idx |= (ptr_5 < v_t)  ? 0x00000020 : 0x00000000;
                        //                    dark_idx |= (ptr_6 < v_t)  ? 0x00000040 : 0x00000000;
                        //                    dark_idx |= (ptr_7 < v_t)  ? 0x00000080 : 0x00000000;
                        //                    dark_idx |= (ptr_8 < v_t)  ? 0x00000100 : 0x00000000;
                        //                    dark_idx |= (ptr_9 < v_t)  ? 0x00000200 : 0x00000000;
                        //                    dark_idx |= (ptr_10 < v_t) ? 0x00000400 : 0x00000000;
                        //                    dark_idx |= (ptr_11 < v_t) ? 0x00000800 : 0x00000000;
                        //                    dark_idx |= (ptr_12 < v_t) ? 0x00001000 : 0x00000000;
                        //                    dark_idx |= (ptr_13 < v_t) ? 0x00002000 : 0x00000000;
                        //                    dark_idx |= (ptr_14 < v_t) ? 0x00004000 : 0x00000000;
                        //                    dark_idx |= (ptr_15 < v_t) ? 0x00008000 : 0x00000000;


                        if(ptr_0 > vt)  bright_idx  = 0x00000001;
                        if(ptr_1 > vt)  bright_idx |= 0x00000002;
                        if(ptr_2 > vt)  bright_idx |= 0x00000004;
                        if(ptr_3 > vt)  bright_idx |= 0x00000008;
                        if(ptr_4 > vt)  bright_idx |= 0x00000010;
                        if(ptr_5 > vt)  bright_idx |= 0x00000020;
                        if(ptr_6 > vt)  bright_idx |= 0x00000040;
                        if(ptr_7 > vt)  bright_idx |= 0x00000080;
                        if(ptr_8 > vt)  bright_idx |= 0x00000100;
                        if(ptr_9 > vt)  bright_idx |= 0x00000200;
                        if(ptr_10 > vt) bright_idx |= 0x00000400;
                        if(ptr_11 > vt) bright_idx |= 0x00000800;
                        if(ptr_12 > vt) bright_idx |= 0x00001000;
                        if(ptr_13 > vt) bright_idx |= 0x00002000;
                        if(ptr_14 > vt) bright_idx |= 0x00004000;
                        if(ptr_15 > vt) bright_idx |= 0x00008000;


                        if(ptr_0 < v_t)  dark_idx  = 0x00000001;
                        if(ptr_1 < v_t)  dark_idx |= 0x00000002;
                        if(ptr_2 < v_t)  dark_idx |= 0x00000004;
                        if(ptr_3 < v_t)  dark_idx |= 0x00000008;
                        if(ptr_4 < v_t)  dark_idx |= 0x00000010;
                        if(ptr_5 < v_t)  dark_idx |= 0x00000020;
                        if(ptr_6 < v_t)  dark_idx |= 0x00000040;
                        if(ptr_7 < v_t)  dark_idx |= 0x00000080;
                        if(ptr_8 < v_t)  dark_idx |= 0x00000100;
                        if(ptr_9 < v_t)  dark_idx |= 0x00000200;
                        if(ptr_10 < v_t) dark_idx |= 0x00000400;
                        if(ptr_11 < v_t) dark_idx |= 0x00000800;
                        if(ptr_12 < v_t) dark_idx |= 0x00001000;
                        if(ptr_13 < v_t) dark_idx |= 0x00002000;
                        if(ptr_14 < v_t) dark_idx |= 0x00004000;
                        if(ptr_15 < v_t) dark_idx |= 0x00008000;

                        //#pragma unroll
                        //                    for(int i=0;i<16;i++)
                        //                    {
                        ////                        bright_idx |= (ptr_arr[i] > vt); bright_idx <<= 1;
                        ////                        dark_idx |= (ptr_arr[i] < v_t); dark_idx <<= 1;

                        ////                        bright_idx |= (ptr_arr[i] > vt) << i;
                        ////                        dark_idx |= (ptr_arr[i] < v_t) << i;

                        ////                        bright_idx += (ptr_arr[i] > vt) ? 1 << i : 0;
                        ////                        dark_idx += (ptr_arr[i] < v_t) ? 1 << i : 0;

                        //                        bright_idx += signbit((float)(-ptr_arr[i] + vt)) ? 1 << i : 0;
                        //                        dark_idx += signbit((float)(ptr_arr[i] - v_t)) ? 1 << i : 0;
                        //                    }

                        if(lookup_table[bright_idx] || lookup_table[dark_idx])
                        {

                            //#pragma unroll
                            //                        for(int i=0;i<16;i++)
                            //                        {
                            //                            local_score += fabsf(ptr_arr[i] - v);
                            //                        }

                            local_score  = fabsf(ptr_0 - v) + fabsf(ptr_1 - v)
                                    + fabsf(ptr_2 - v) + fabsf(ptr_3 - v)
                                    + fabsf(ptr_4 - v) + fabsf(ptr_5 - v)
                                    + fabsf(ptr_6 - v) + fabsf(ptr_7 - v)
                                    + fabsf(ptr_8 - v) + fabsf(ptr_9 - v)
                                    + fabsf(ptr_10 - v) + fabsf(ptr_11 - v)
                                    + fabsf(ptr_12 - v) + fabsf(ptr_13 - v)
                                    + fabsf(ptr_14 - v) + fabsf(ptr_15 - v);
                        }

                    }
                }

                score_data[score_offset] = local_score;

            }
        }

    }
}



void ORB_GPU::FAST_compute_score_lookpup(int height, int width,
                                         unsigned char* image_data_gpu,
                                         int image_pitch,
                                         int threshold,
                                         int* lookup_table_gpu,
                                         int* score_data_gpu,
                                         int score_pitch,
                                         cudaStream_t& cuda_stream)
{

#ifdef GRID_LAUNCH

    {
        int CUDA_NUM_BLOCKS_x = (width  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
        int CUDA_NUM_BLOCKS_y = (height - 1) / CUDA_NUM_THREADS_PER_BLOCK_y + 1;

        dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
        dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, CUDA_NUM_THREADS_PER_BLOCK_y, 1);


        FASTComputeScoreGPU_patternSize_16_lookup<<<grid_dim, block_dim,0 , cuda_stream>>>(
                                                                                             height, width,
                                                                                             threshold,
                                                                                             lookup_table_gpu,
                                                                                             image_data_gpu,
                                                                                             image_pitch,
                                                                                             score_data_gpu,
                                                                                             score_pitch);

    }

#else
    // slower 10-12us
    {
        int roiheight = (height - 2*BORDER_SKIP);
        int roiwidth  = (width - 2*BORDER_SKIP);

        int n_threads =  roiheight * roiwidth;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1 ;

        FASTComputeScoreGPU_patternSize_16<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                              n_threads,
                                                                                                              height, width,
                                                                                                              roiheight, roiwidth,
                                                                                                              threshold_,
                                                                                                              image_data_gpu,
                                                                                                              score_data_gpu);

    }

#endif



}




__global__ void FASTComputeScoreGPU_patternSize_16_lookup_mask(int imheight, int imwidth,
                                                               int threshold,
                                                               int* lookup_table,
                                                               const unsigned char* image_data,
                                                               int image_pitch,
                                                               const unsigned char* mask_data,
                                                               int mask_pitch,
                                                               int* score_data,
                                                               int score_pitch)
{

#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;


    if(w < imwidth && h < imheight)
    {
        const int h_im = h;// + BORDER_SKIP;
        const int w_im = w;// + BORDER_SKIP;

        const int offset = h_im * image_pitch + w_im;
        const int score_offset = h_im * score_pitch + w_im;

        if(!mask_data[h_im * mask_pitch + w_im])
            return;

#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int h = index / roiwidth;
        int w = index % roiwidth;
#endif

        if(w >= BORDER_SKIP && w < imwidth - BORDER_SKIP && h < imheight - BORDER_SKIP && h >= BORDER_SKIP)
        {

            int local_score = 0;

            const unsigned char* ptr = image_data + offset;

            const int v = ptr[0];

            const int vt = v + threshold;
            const int v_t = v - threshold;


            const int ptr_4  = ptr[3];
            const int ptr_12 = ptr[-3];


            if(ptr_4 <= vt && ptr_4 >= v_t && ptr_12 <= vt && ptr_12 >= v_t)
            {        score_data[score_offset] =  0;  return;}


            const int imw3 = 3*image_pitch;
            const int nimw3 = -imw3;

            const int ptr_0 = ptr[imw3];
            const int ptr_8 = ptr[nimw3];

            if(ptr_0 <= vt && ptr_0 >= v_t && ptr_8 <= vt && ptr_8 >= v_t)
            {        score_data[score_offset] =  0;return;}


            const int imw2 = 2*image_pitch;
            const int nimw2 = -imw2;
            const int nimw = -image_pitch;

            const int ptr_1  = ptr[imw3+1];
            const int ptr_2  = ptr[imw2+2];
            const int ptr_3  = ptr[image_pitch+3];
            const int ptr_5  = ptr[nimw+3];
            const int ptr_6  = ptr[nimw2+2];
            const int ptr_7  = ptr[nimw3+1];
            const int ptr_9  = ptr[nimw3-1];
            const int ptr_10 = ptr[nimw2-2];
            const int ptr_11 = ptr[nimw-3];
            const int ptr_13 = ptr[image_pitch-3];
            const int ptr_14 = ptr[imw2-2];
            const int ptr_15 = ptr[imw3-1];

            {
                {
                    int bright_idx = 0;
                    int dark_idx = 0;

                    {

                        if(ptr_0 > vt)  bright_idx  = 0x00000001;
                        if(ptr_1 > vt)  bright_idx |= 0x00000002;
                        if(ptr_2 > vt)  bright_idx |= 0x00000004;
                        if(ptr_3 > vt)  bright_idx |= 0x00000008;
                        if(ptr_4 > vt)  bright_idx |= 0x00000010;
                        if(ptr_5 > vt)  bright_idx |= 0x00000020;
                        if(ptr_6 > vt)  bright_idx |= 0x00000040;
                        if(ptr_7 > vt)  bright_idx |= 0x00000080;
                        if(ptr_8 > vt)  bright_idx |= 0x00000100;
                        if(ptr_9 > vt)  bright_idx |= 0x00000200;
                        if(ptr_10 > vt) bright_idx |= 0x00000400;
                        if(ptr_11 > vt) bright_idx |= 0x00000800;
                        if(ptr_12 > vt) bright_idx |= 0x00001000;
                        if(ptr_13 > vt) bright_idx |= 0x00002000;
                        if(ptr_14 > vt) bright_idx |= 0x00004000;
                        if(ptr_15 > vt) bright_idx |= 0x00008000;


                        if(ptr_0 < v_t)  dark_idx  = 0x00000001;
                        if(ptr_1 < v_t)  dark_idx |= 0x00000002;
                        if(ptr_2 < v_t)  dark_idx |= 0x00000004;
                        if(ptr_3 < v_t)  dark_idx |= 0x00000008;
                        if(ptr_4 < v_t)  dark_idx |= 0x00000010;
                        if(ptr_5 < v_t)  dark_idx |= 0x00000020;
                        if(ptr_6 < v_t)  dark_idx |= 0x00000040;
                        if(ptr_7 < v_t)  dark_idx |= 0x00000080;
                        if(ptr_8 < v_t)  dark_idx |= 0x00000100;
                        if(ptr_9 < v_t)  dark_idx |= 0x00000200;
                        if(ptr_10 < v_t) dark_idx |= 0x00000400;
                        if(ptr_11 < v_t) dark_idx |= 0x00000800;
                        if(ptr_12 < v_t) dark_idx |= 0x00001000;
                        if(ptr_13 < v_t) dark_idx |= 0x00002000;
                        if(ptr_14 < v_t) dark_idx |= 0x00004000;
                        if(ptr_15 < v_t) dark_idx |= 0x00008000;

                        if(lookup_table[bright_idx] || lookup_table[dark_idx])
                        {
                            local_score  = fabsf(ptr_0 - v) + fabsf(ptr_1 - v)
                                    + fabsf(ptr_2 - v) + fabsf(ptr_3 - v)
                                    + fabsf(ptr_4 - v) + fabsf(ptr_5 - v)
                                    + fabsf(ptr_6 - v) + fabsf(ptr_7 - v)
                                    + fabsf(ptr_8 - v) + fabsf(ptr_9 - v)
                                    + fabsf(ptr_10 - v) + fabsf(ptr_11 - v)
                                    + fabsf(ptr_12 - v) + fabsf(ptr_13 - v)
                                    + fabsf(ptr_14 - v) + fabsf(ptr_15 - v);
                        }

                    }
                }

                score_data[score_offset] = local_score;

            }
        }

    }
}



void ORB_GPU::FAST_compute_score_lookpup_mask(int height, int width,
                                              unsigned char* image_data_gpu,
                                              int image_pitch,
                                              unsigned char* mask_data_gpu,
                                              int mask_pitch,
                                              int threshold,
                                              int* lookup_table_gpu,
                                              int* score_data_gpu,
                                              int score_pitch,
                                              cudaStream_t& cuda_stream)
{

#ifdef GRID_LAUNCH

    {
        int CUDA_NUM_BLOCKS_x = (width  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
        int CUDA_NUM_BLOCKS_y = (height - 1) / CUDA_NUM_THREADS_PER_BLOCK_y + 1;

        dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
        dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, CUDA_NUM_THREADS_PER_BLOCK_y, 1);


        FASTComputeScoreGPU_patternSize_16_lookup_mask<<<grid_dim, block_dim,0 , cuda_stream>>>(
                                                                                                  height, width,
                                                                                                  threshold,
                                                                                                  lookup_table_gpu,
                                                                                                  image_data_gpu,
                                                                                                  image_pitch,
                                                                                                  mask_data_gpu,
                                                                                                  mask_pitch,
                                                                                                  score_data_gpu,
                                                                                                  score_pitch);

    }

#else
    // slower 10-12us
    {
        int roiheight = (height - 2*BORDER_SKIP);
        int roiwidth  = (width - 2*BORDER_SKIP);

        int n_threads =  roiheight * roiwidth;

        int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1 ;

        FASTComputeScoreGPU_patternSize_16<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                              n_threads,
                                                                                                              height, width,
                                                                                                              roiheight, roiwidth,
                                                                                                              threshold_,
                                                                                                              image_data_gpu,
                                                                                                              score_data_gpu);

    }

#endif



}

}





//__global__ void FASTComputeScoreGPU(int n_threads,
//                                    int imheight, int imwidth,
//                                    int roiheight, int roiwidth,
//                                    int threshold,
//                                    int patternSize, int* pixel_circle,
//                                    int* threshold_tab,
//                                    unsigned char* image_data,
//                                    int* score_data)
//{
//    int index = blockIdx.x * blockDim.x + threadIdx.x;

//    if(index <  n_threads)
//    {
//        int h = index / roiwidth;
//        int w = index % roiwidth;

//        int h_im = h + BORDER_SKIP;
//        int w_im = w + BORDER_SKIP;

//        int offset = h_im * imwidth + w_im;

//        const unsigned char* ptr = image_data + offset;

//        int v = ptr[0];
//        int neg_threshold = -threshold;

//        int n_bright = 0;
//        int n_dark = 0;

//        int bright_sum = 0 ;
//        int dark_sum = 0;


//        {
//            int local_score = 0;

//            const int* tab = &threshold_tab[0] - v + 255;
//            int d = tab[ptr[pixel_circle[0]]] | tab[ptr[pixel_circle[8]]];

//            if(d)
//            {
//                d &= tab[ptr[pixel_circle[2]]] | tab[ptr[pixel_circle[10]]];
//                d &= tab[ptr[pixel_circle[4]]] | tab[ptr[pixel_circle[12]]];
//                d &= tab[ptr[pixel_circle[6]]] | tab[ptr[pixel_circle[14]]];

//                if(d)
//                {
//                    d &= tab[ptr[pixel_circle[1]]] | tab[ptr[pixel_circle[9]]];
//                    d &= tab[ptr[pixel_circle[3]]] | tab[ptr[pixel_circle[11]]];
//                    d &= tab[ptr[pixel_circle[5]]] | tab[ptr[pixel_circle[13]]];
//                    d &= tab[ptr[pixel_circle[7]]] | tab[ptr[pixel_circle[15]]];


//                    if(d)
//                    {

//                        //                        // minimal divergence
//                        //                        {
//                        //                            for(int m=0;m<patternSize;m++)
//                        //                            {
//                        //                                int x = ptr[pixel_circle[m]] - v;
//                        //                                if(x > threshold)
//                        //                                {
//                        //                                    n_bright++;
//                        //                                    bright_sum += x;
//                        //                                }
//                        //                                else if(x < neg_threshold)
//                        //                                {
//                        //                                    n_dark++;
//                        //                                    dark_sum += -x;
//                        //                                }
//                        //                            }
//                        //                            int min_num = patternSize * 0.95;
//                        //                            if(n_bright >= min_num || n_dark >= min_num)
//                        //                                local_score = (bright_sum > dark_sum) ? bright_sum : dark_sum; // this sum of absolute differences, already there in the literature

//                        //                            //                            int min_num = patternSize * 0.7;
//                        //                            //                            int max_num = patternSize * 0.8;

//                        //                            //                            if((n_bright >= min_num && n_bright <=max_num) || (n_dark >= min_num && n_dark <= max_num))
//                        //                            //                                local_score = (bright_sum > dark_sum) ? bright_sum : dark_sum; // this sum of absolute differences, already there in the literature

//                        //                            //                            if((n_bright >= min_num && n_bright <=max_num) || (n_dark >= min_num && n_dark <= max_num))
//                        //                            //                                local_score = (bright_sum > dark_sum) ? (n_bright *bright_sum) : (n_dark*dark_sum); // this sum of absolute differences, already there in the literature

//                        //                            ///proposed in the paper
//                        //                            //   local_score = min_num;
//                        //                        }

//                        //  divergence less kernel with new score
//                        {
//                            for(int m=0;m<patternSize;m++)
//                            {
//                                int x = ptr[pixel_circle[m]] - v;
//                                int res1 = x > threshold;
//                                int res2 = x < neg_threshold;

//                                n_bright += res1;
//                                n_dark += res2;

//                                bright_sum += res1 * x;
//                                dark_sum += res2 * (-x);
//                            }

//                            int min_num = patternSize * 0.60;
//                            int max_num = patternSize * 0.9;

//                            int min_bright = n_bright >= min_num;
//                            int min_dark = n_dark >= min_num;

//                            int max_bright = n_bright <= max_num;
//                            int max_dark = n_dark <= max_num;

//                            int in_range_bright = min_bright && max_bright;
//                            int in_range_dark = min_dark && max_dark;

//                            // this sum of absolute differences, already there in the literature
//                            local_score = in_range_bright * n_bright * bright_sum + in_range_dark * n_dark * dark_sum;
//                            //                            local_score = in_range_bright * bright_sum + in_range_dark * dark_sum;

//                            //proposed in the paper
//                            //                            local_score = min_num * (min_bright * more_bright+ min_dark * more_dark);
//                        }


//                        //// no divergence
//                        //                        {
//                        //                            for(int m=0;m<patternSize;m++)
//                        //                            {
//                        //                                int x = ptr[pixel_circle[m]] - v;
//                        //                                int res1 = x > threshold;
//                        //                                int res2 = x < neg_threshold;

//                        //                                n_bright += res1;
//                        //                                n_dark += res2;

//                        //                                bright_sum += res1 * x;
//                        //                                dark_sum += res2 * (-x);
//                        //                            }

//                        //                            int min_num = patternSize * 0.95;

//                        //                            int min_bright = n_bright >= min_num;
//                        //                            int min_dark = n_dark >= min_num;

//                        //                            int more_bright = bright_sum > dark_sum;
//                        //                            int more_dark = bright_sum <= dark_sum;

//                        //                            // this sum of absolute differences, already there in the literature
//                        //                            local_score = min_bright * more_bright * bright_sum + min_dark * more_dark * dark_sum;

//                        //                            //proposed in the paper
//                        //                            //                            local_score = min_num * (min_bright * more_bright+ min_dark * more_dark);
//                        //                        }


//                        // minimal divergence continuous arc
//                        //                        {
//                        //                            int prev = 0;

//                        //                            for(int m=0;m<patternSize;m++)
//                        //                            {
//                        //                                int x = ptr[pixel_circle[m]] - v;

//                        //                                if(m==0)
//                        //                                {
//                        //                                    if(x > threshold)
//                        //                                        prev = 0;
//                        //                                    else if(x < neg_threshold)
//                        //                                        prev = 1;
//                        //                                }

//                        //                                if(x > threshold)
//                        //                                {
//                        //                                    n_bright++;
//                        //                                    bright_sum += x;

//                        //                                    if(m!=0 && prev == 1)
//                        //                                    {
//                        //                                        n_bright = 0;
//                        //                                        bright_sum = 0;
//                        //                                    }
//                        //                                }
//                        //                                else if(x < neg_threshold)
//                        //                                {
//                        //                                    n_dark++;
//                        //                                    dark_sum += -x;

//                        //                                    if(m!=0 && prev == 0)
//                        //                                    {
//                        //                                        n_dark = 0;
//                        //                                        dark_sum = 0;
//                        //                                    }
//                        //                                }


//                        //                            }

//                        //                            int min_num = patternSize * 0.6;
//                        //                            int max_num = patternSize * 0.9;

//                        //                            int min_bright = n_bright >= min_num;
//                        //                            int min_dark = n_dark >= min_num;

//                        //                            int max_bright = n_bright <= max_num;
//                        //                            int max_dark = n_dark <= max_num;

//                        //                            int in_range_bright = min_bright && max_bright;
//                        //                            int in_range_dark = min_dark && max_dark;

//                        //                            // this sum of absolute differences, already there in the literature
//                        //                            local_score = in_range_bright * bright_sum + in_range_dark * dark_sum;


//                        //                            //                            int min_num = patternSize * 0.95;

//                        //                            //                            if(n_bright >= min_num || n_dark >= min_num)
//                        //                            //                                local_score = (bright_sum > dark_sum) ? bright_sum : dark_sum; // this sum of absolute differences, already there in the literature

//                        //                            //                            ///proposed in the paper
//                        //                            //                            //   local_score = min_num;
//                        //                        }

//                    }
//                }
//            }

//            score_data[offset] = local_score;

//        }
//    }
//}


