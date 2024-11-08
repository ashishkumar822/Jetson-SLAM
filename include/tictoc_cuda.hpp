#ifndef __TICTOC_CUDA_HPP___
#define __TICTOC_CUDA_HPP___

#include<iostream>
#include<chrono>

#include<cuda.h>
#include<cuda_runtime.h>

class tictoc
{
public:

    tictoc()
    {
        cudaEventCreate(&t1);
        cudaEventCreate(&t2);

        stream = NULL;
    }

    ~tictoc()
    {
        cudaEventDestroy(t1);
        cudaEventDestroy(t2);
    }


    void tic(void)
    {
        cudaEventRecord(t1,stream);
    }

    float toc(void)
    {
        cudaEventRecord(t2,stream);
        cudaEventSynchronize(t2);
        cudaEventElapsedTime(&time_ms, t1, t2);
        return time_ms;
    }

    void toc_print(void)
    {
        cudaEventRecord(t2,stream);
        cudaEventSynchronize(t2);
        cudaEventElapsedTime(&time_ms, t1, t2);
        std::cout << "Time elapsed = " << time_ms << " ms\n";
    }

    void print(void)
    {
        std::cout << "Time elapsed = " << time_ms << " ms\n";
    }

    float time(void)
    {
        return time_ms;
    }

    cudaStream_t stream;

private:
    cudaEvent_t t1;
    cudaEvent_t t2;

    float time_ms;
};

#endif
