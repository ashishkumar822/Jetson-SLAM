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



#ifndef __SYNCED_MEM_HOLDER_HPP__
#define __SYNCED_MEM_HOLDER_HPP__

#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuda_device_runtime_api.h>

namespace orb_cuda {

template <typename Dtype>
class SyncedMem {

public:

    SyncedMem(void);

    ~SyncedMem();


    void resize(int count);

    Dtype* cpu_data();
    Dtype* gpu_data();

    void to_cpu(void);
    void to_gpu(void);

    void to_cpu(int count);
    void to_gpu(int count);

    void to_cpu_async(void);
    void to_gpu_async(void);

    void to_cpu_async(cudaStream_t& cu_stream);
    void to_gpu_async(cudaStream_t& cu_stream);

    void to_cpu_async(int count);
    void to_gpu_async(int count);

    void to_cpu_async(cudaStream_t& cu_stream, int count);
    void to_gpu_async(cudaStream_t& cu_stream, int count);

    void resize_pitched(size_t width, size_t height);

    void sync_stream(void);

    void set_zero_gpu(void);
    void set_zero_gpu_async(void);

    void set_zero_cpu(void);

//private:

    int count_;
    int capacity_;

    Dtype* cpu_data_;
    Dtype* gpu_data_;

    size_t pitch_;

    cudaStream_t cu_stream_;
    cudaError_t cu_error_;

};

}

#endif
