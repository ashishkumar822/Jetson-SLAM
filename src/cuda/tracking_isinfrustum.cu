#include<cuda/tracking_gpu.hpp>

#ifdef STATIC_MEM_IS_IN

#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<stdio.h>

#include<iostream>

#include<chrono>

namespace tracking_cuda{

#define CUDA_NUM_THREADS_PER_BLOCK 512

__global__ void isInFrustum_GPU(int n_threads,
                                float* Px, float* Py, float* Pz,
                                float* Pnx, float* Pny, float* Pnz,
                                float* MaxDistance,
                                float* invariance_maxDistance,
                                float* invariance_minDistance,
                                float* Rcw, float* tcw, float* Ow,
                                float fx, float fy, float cx, float cy,
                                int minX, int maxX, int minY, int maxY,
                                int nScaleLevels,
                                float logScaleFactor,
                                float viewCosAngle,
                                float* invz, float* u, float* v,
                                int* predictedlevel,
                                float* viewCos,
                                unsigned char* is_infrustum)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        unsigned char infrustum = 0;

        const float Pwx = Px[index];
        const float Pwy = Py[index];
        const float Pwz = Pz[index];

        const float Pcx = Pwx * Rcw[0] + Pwy * Rcw[1] + Pwz * Rcw[2] + tcw[0];
        const float Pcy = Pwx * Rcw[3] + Pwy * Rcw[4] + Pwz * Rcw[5] + tcw[1];
        const float Pcz = Pwx * Rcw[6] + Pwy * Rcw[7] + Pwz * Rcw[8] + tcw[2];

        // Check positive depth
        if(Pcz > 0.0f)
        {
            // Project in image and check it is not outside
            const float im_invz = 1.0f / Pcz;
            const float im_u = fx * Pcx * im_invz + cx;
            const float im_v = fy * Pcy * im_invz + cy;

            if(!(im_u < minX || im_u > maxX || im_v < minY || im_v > maxY))
            {
                const float maxDistance = invariance_maxDistance[index];
                const float minDistance = invariance_minDistance[index];

                const float Pox = Pwx - Ow[0];
                const float Poy = Pwy - Ow[1];
                const float Poz = Pwz - Ow[2];

                const float dist = sqrtf(Pox*Pox + Poy*Poy + Poz*Poz);

                if(!(dist<minDistance || dist>maxDistance))
                {

                    const float Pwnx = Pnx[index];
                    const float Pwny = Pny[index];
                    const float Pwnz = Pnz[index];

                    const float p_viewCos = (Pox*Pwnx + Poy*Pwny + Poz*Pwnz) / dist;

                    if(!(p_viewCos < viewCosAngle))
                    {
                        float ratio = MaxDistance[index] / dist;

                        int nScale = ceilf(logf(ratio)/logScaleFactor);

                        if(nScale < 0)
                        {
                            nScale = 0;
                        }
                        else if(nScale >= nScaleLevels)
                        {
                            nScale = nScaleLevels-1;
                        }

                        u[index] = im_u;
                        v[index] = im_v;
                        invz[index] = im_invz;
                        predictedlevel[index] = nScale;
                        viewCos[index] = p_viewCos;

                        infrustum = 1;

                    }
                }
            }
        }

        is_infrustum[index] = infrustum;
    }
}


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
                             unsigned char* is_infrustum_gpu)
{

    int n_threads = n_points;

    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

    isInFrustum_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK>>>(
                                                                       n_threads,
                                                                       Px_gpu, Py_gpu, Pz_gpu,
                                                                       Pnx_gpu, Pny_gpu, Pnz_gpu,
                                                                       MaxDistance_gpu,
                                                                       invariance_maxDistance_gpu,
                                                                       invariance_minDistance_gpu,
                                                                       Rcw_gpu, tcw_gpu, Ow_gpu,
                                                                       fx, fy, cx, cy,
                                                                       minX, maxX, minY, maxY,
                                                                       nScaleLevels,
                                                                       logScaleFactor,
                                                                       viewCosAngle,
                                                                       invz_gpu, u_gpu,  v_gpu,
                                                                       predictedlevel_gpu,
                                                                       viewCos_gpu,
                                                                       is_infrustum_gpu
                                                                       );

    cudaStreamSynchronize(0);
}


}


#else

#include<cuda/tracking_gpu.hpp>

#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<stdio.h>

#include<iostream>

#include<chrono>

namespace tracking_cuda{

#define CUDA_NUM_THREADS_PER_BLOCK 512

__global__ void isInFrustum_GPU(int n_threads,
                                float* Px, float* Py, float* Pz,
                                float* Pnx, float* Pny, float* Pnz,
                                float* MaxDistance,
                                float* invariance_maxDistance,
                                float* invariance_minDistance,
                                float* Rcw, float* tcw, float* Ow,
                                float fx, float fy, float cx, float cy,
                                int minX, int maxX, int minY, int maxY,
                                int nScaleLevels,
                                float logScaleFactor,
                                float viewCosAngle,
                                float* invz, float* u, float* v,
                                int* predictedlevel,
                                float* viewCos,
                                unsigned char* is_infrustum)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        unsigned char infrustum = 0;

        const float Pwx = Px[index];
        const float Pwy = Py[index];
        const float Pwz = Pz[index];

        const float Pcx = Pwx * Rcw[0] + Pwy * Rcw[1] + Pwz * Rcw[2] + tcw[0];
        const float Pcy = Pwx * Rcw[3] + Pwy * Rcw[4] + Pwz * Rcw[5] + tcw[1];
        const float Pcz = Pwx * Rcw[6] + Pwy * Rcw[7] + Pwz * Rcw[8] + tcw[2];

        // Check positive depth
        if(Pcz > 0.0f)
        {
            // Project in image and check it is not outside
            const float im_invz = 1.0f / Pcz;
            const float im_u = fx * Pcx * im_invz + cx;
            const float im_v = fy * Pcy * im_invz + cy;

            if(!(im_u < minX || im_u > maxX || im_v < minY || im_v > maxY))
            {
                const float maxDistance = invariance_maxDistance[index];
                const float minDistance = invariance_minDistance[index];

                const float Pox = Pwx - Ow[0];
                const float Poy = Pwy - Ow[1];
                const float Poz = Pwz - Ow[2];

                const float dist = sqrtf(Pox*Pox + Poy*Poy + Poz*Poz);

                if(!(dist<minDistance || dist>maxDistance))
                {

                    const float Pwnx = Pnx[index];
                    const float Pwny = Pny[index];
                    const float Pwnz = Pnz[index];

                    const float p_viewCos = (Pox*Pwnx + Poy*Pwny + Poz*Pwnz) / dist;

                    if(!(p_viewCos < viewCosAngle))
                    {
                        float ratio = MaxDistance[index] / dist;

                        int nScale = ceilf(logf(ratio)/logScaleFactor);

                        if(nScale < 0)
                        {
                            nScale = 0;
                        }
                        else if(nScale >= nScaleLevels)
                        {
                            nScale = nScaleLevels-1;
                        }

                        u[index] = im_u;
                        v[index] = im_v;
                        invz[index] = im_invz;
                        predictedlevel[index] = nScale;
                        viewCos[index] = p_viewCos;

                        infrustum = 1;

                    }
                }
            }
        }

        is_infrustum[index] = infrustum;
    }
}


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
                             std::vector<unsigned char>& is_infrustum)
{

    float* Px_gpu;
    float* Py_gpu;
    float* Pz_gpu;

    float* Pnx_gpu;
    float* Pny_gpu;
    float* Pnz_gpu;

    float* MaxDistance_gpu;
    float* invariance_maxDistance_gpu;
    float* invariance_minDistance_gpu;

    float* Rcw_gpu;
    float* tcw_gpu;
    float* Ow_gpu;

    float* invz_gpu;
    float* u_gpu;
    float* v_gpu;

    int* predictedlevel_gpu;
    float* viewCos_gpu;

    unsigned char* is_infrustum_gpu;

    int n_points = Px.size();


    cudaMalloc((void**)&Px_gpu, sizeof(float)*n_points);
    cudaMalloc((void**)&Py_gpu, sizeof(float)*n_points);
    cudaMalloc((void**)&Pz_gpu, sizeof(float)*n_points);

    cudaMalloc((void**)&Pnx_gpu, sizeof(float)*n_points);
    cudaMalloc((void**)&Pny_gpu, sizeof(float)*n_points);
    cudaMalloc((void**)&Pnz_gpu, sizeof(float)*n_points);

    cudaMalloc((void**)&MaxDistance_gpu, sizeof(float)*n_points);
    cudaMalloc((void**)&invariance_maxDistance_gpu, sizeof(float)*n_points);
    cudaMalloc((void**)&invariance_minDistance_gpu, sizeof(float)*n_points);

    cudaMalloc((void**)&Rcw_gpu, sizeof(float)*9);
    cudaMalloc((void**)&tcw_gpu, sizeof(float)*3);
    cudaMalloc((void**)&Ow_gpu, sizeof(float)*3);


    cudaMalloc((void**)&invz_gpu, sizeof(float)*n_points);
    cudaMalloc((void**)&u_gpu, sizeof(float)*n_points);
    cudaMalloc((void**)&v_gpu, sizeof(float)*n_points);

    cudaMalloc((void**)&predictedlevel_gpu, sizeof(int)*n_points);
    cudaMalloc((void**)&viewCos_gpu, sizeof(float)*n_points);

    cudaMalloc((void**)&is_infrustum_gpu, sizeof(unsigned char)*n_points);


    cudaMemcpy(Px_gpu, Px.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);
    cudaMemcpy(Py_gpu, Py.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);
    cudaMemcpy(Pz_gpu, Pz.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);

    cudaMemcpy(Pnx_gpu, Pnx.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);
    cudaMemcpy(Pny_gpu, Pny.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);
    cudaMemcpy(Pnz_gpu, Pnz.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);

    cudaMemcpy(MaxDistance_gpu, MaxDistance.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);
    cudaMemcpy(invariance_maxDistance_gpu, invariance_maxDistance.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);
    cudaMemcpy(invariance_minDistance_gpu, invariance_minDistance.data(), sizeof(float)*n_points, cudaMemcpyHostToDevice);

    cudaMemcpy(Rcw_gpu, Rcw.data(), sizeof(float)*9, cudaMemcpyHostToDevice);
    cudaMemcpy(tcw_gpu, tcw.data(), sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(Ow_gpu, Ow.data(), sizeof(float)*3, cudaMemcpyHostToDevice);

    {

        int n_threads = n_points;

        int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

        isInFrustum_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK>>>(
                                                                           n_threads,
                                                                           Px_gpu, Py_gpu, Pz_gpu,
                                                                           Pnx_gpu, Pny_gpu, Pnz_gpu,
                                                                           MaxDistance_gpu,
                                                                           invariance_maxDistance_gpu,
                                                                           invariance_minDistance_gpu,
                                                                           Rcw_gpu, tcw_gpu, Ow_gpu,
                                                                           fx, fy, cx, cy,
                                                                           minX, maxX, minY, maxY,
                                                                           nScaleLevels,
                                                                           logScaleFactor,
                                                                           viewCosAngle,
                                                                           invz_gpu, u_gpu,  v_gpu,
                                                                           predictedlevel_gpu,
                                                                           viewCos_gpu,
                                                                           is_infrustum_gpu
                                                                           );

        cudaStreamSynchronize(0);
    }


    cudaMemcpy(invz.data(), invz_gpu, sizeof(float)*n_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(u.data(), u_gpu, sizeof(float)*n_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), v_gpu, sizeof(float)*n_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(predictedlevel.data(), predictedlevel_gpu, sizeof(int)*n_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(viewCos.data(), viewCos_gpu, sizeof(float)*n_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(is_infrustum.data(), is_infrustum_gpu, sizeof(unsigned char)*n_points, cudaMemcpyDeviceToHost);


    cudaFree(Px_gpu);
    cudaFree(Py_gpu);
    cudaFree(Pz_gpu);

    cudaFree(Pnx_gpu);
    cudaFree(Pny_gpu);
    cudaFree(Pnz_gpu);

    cudaFree(MaxDistance_gpu);
    cudaFree(invariance_maxDistance_gpu);
    cudaFree(invariance_minDistance_gpu);

    cudaFree(Rcw_gpu);
    cudaFree(tcw_gpu);
    cudaFree(Ow_gpu);


    cudaFree(invz_gpu);
    cudaFree(u_gpu);
    cudaFree(v_gpu);

    cudaFree(predictedlevel_gpu);
    cudaFree(viewCos_gpu);

    cudaFree(is_infrustum_gpu);
}


}


#endif
