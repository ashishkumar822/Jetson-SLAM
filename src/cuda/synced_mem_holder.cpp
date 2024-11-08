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

// Part of the middle-end

#include "cuda/synced_mem_holder.hpp"

#include<iostream>
#include<string.h>

namespace orb_cuda {

template<typename Dtype>
SyncedMem<Dtype>::SyncedMem(void)
{
    count_ = 0;
    capacity_ = 0;
    cpu_data_ = 0;
    gpu_data_ = 0;
    cu_stream_ = NULL;
    pitch_ = 0;

    cudaStreamCreate(&cu_stream_);
}

template<typename Dtype>
SyncedMem<Dtype>::~SyncedMem() {

    //finish off anything going on in cu_stream_
    cudaStreamSynchronize(cu_stream_);
    cudaStreamDestroy(cu_stream_);

    cu_stream_ = NULL;
    if(cpu_data_)
        cudaFreeHost(cpu_data_);

    if(gpu_data_)
        cudaFree(gpu_data_);
}


template<typename Dtype>
void SyncedMem<Dtype>::resize_pitched(size_t width, size_t height)
{
    count_ = width * height;

    int capacity_ = count_;

    if(cpu_data_)
        cudaFreeHost(cpu_data_);
    cudaMallocHost((void**)&cpu_data_, capacity_ * sizeof(Dtype));

    if(gpu_data_)
        cudaFree(gpu_data_);
    cudaMallocPitch((void**)&gpu_data_, &pitch_, width * sizeof(Dtype), height);

}

template<typename Dtype>
void SyncedMem<Dtype>::resize(int count)
{
    count_ = count;

    if(capacity_ < count_)
    {
        capacity_ = count_;

        if(cpu_data_)
            cudaFreeHost(cpu_data_);
        cudaMallocHost((void**)&cpu_data_, capacity_ * sizeof(Dtype));

        if(gpu_data_)
            cudaFree(gpu_data_);
        cudaMalloc((void**)&gpu_data_, capacity_ * sizeof(Dtype));

    }
}


template<typename Dtype>
Dtype* SyncedMem<Dtype>::cpu_data()
{
    return cpu_data_;
}

template<typename Dtype>
Dtype* SyncedMem<Dtype>::gpu_data()
{
    return gpu_data_;
}


template<typename Dtype>
void SyncedMem<Dtype>::to_cpu(void)
{
    cudaMemcpy(cpu_data_, gpu_data_, count_ * sizeof(Dtype), cudaMemcpyDeviceToHost);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu(void)
{
    cudaMemcpy(gpu_data_, cpu_data_, count_ * sizeof(Dtype), cudaMemcpyHostToDevice);
}


template<typename Dtype>
void SyncedMem<Dtype>::set_zero_cpu(void)
{
    memset(cpu_data_, 0, count_ * sizeof(Dtype));
}

template<typename Dtype>
void SyncedMem<Dtype>::set_zero_gpu(void)
{
    cudaMemset(gpu_data_, 0, count_ * sizeof(Dtype));
}

template<typename Dtype>
void SyncedMem<Dtype>::set_zero_gpu_async(void)
{
    cudaMemsetAsync(gpu_data_, 0, count_ * sizeof(Dtype), cu_stream_);
}


template<typename Dtype>
void SyncedMem<Dtype>::to_cpu_async(void)
{
    cudaMemcpyAsync(cpu_data_, gpu_data_, count_ * sizeof(Dtype), cudaMemcpyDeviceToHost, cu_stream_);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu_async(void)
{
    cudaMemcpyAsync(gpu_data_, cpu_data_, count_ * sizeof(Dtype), cudaMemcpyHostToDevice, cu_stream_);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_cpu_async(cudaStream_t& cu_stream)
{
    cudaMemcpyAsync(cpu_data_, gpu_data_, count_ * sizeof(Dtype), cudaMemcpyDeviceToHost, cu_stream);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu_async(cudaStream_t& cu_stream)
{
    cudaMemcpyAsync(gpu_data_, cpu_data_, count_ * sizeof(Dtype), cudaMemcpyHostToDevice, cu_stream);
}


// desired count transfer CPU-GPU
template<typename Dtype>
void SyncedMem<Dtype>::to_cpu(int count)
{
    cudaMemcpy(cpu_data_, gpu_data_, count * sizeof(Dtype), cudaMemcpyDeviceToHost);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu(int count)
{
    cudaMemcpy(gpu_data_, cpu_data_, count * sizeof(Dtype), cudaMemcpyHostToDevice);
}



template<typename Dtype>
void SyncedMem<Dtype>::to_cpu_async(int count)
{
    cudaMemcpyAsync(cpu_data_, gpu_data_, count * sizeof(Dtype), cudaMemcpyDeviceToHost, cu_stream_);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu_async(int count)
{
    cudaMemcpyAsync(gpu_data_, cpu_data_, count * sizeof(Dtype), cudaMemcpyHostToDevice, cu_stream_);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_cpu_async(cudaStream_t& cu_stream,int count)
{
    cudaMemcpyAsync(cpu_data_, gpu_data_, count * sizeof(Dtype), cudaMemcpyDeviceToHost, cu_stream);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu_async(cudaStream_t& cu_stream,int count)
{
    cudaMemcpyAsync(gpu_data_, cpu_data_, count * sizeof(Dtype), cudaMemcpyHostToDevice, cu_stream);
}



template<typename Dtype>
void SyncedMem<Dtype>::sync_stream(void)
{
    cudaStreamSynchronize(cu_stream_);
}

template class SyncedMem<unsigned char>;
template class SyncedMem<signed char>;
template class SyncedMem<unsigned short>;
template class SyncedMem<short>;
template class SyncedMem<unsigned int>;
template class SyncedMem<int>;
template class SyncedMem<float>;


}




