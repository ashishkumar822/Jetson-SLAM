#ifndef __DJI_TICTOC_HPP___
#define __DJI_TICTOC_HPP___

#include<iostream>
#include<chrono>

class tictoc
{
public:
    void tic(void)
    {
        t1 = std::chrono::steady_clock::now();
    }

    float toc(void)
    {
        t2 = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() * 1000.0;
        return time_ms;
    }

    void toc_print(void)
    {
        t2 = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() * 1000.0;
        std::cout << time_ms << " ms\n";
    }

    void print(void)
    {
        std::cout << time_ms << " ms\n";
    }

    float time(void)
    {
        return time_ms;
    }

private:
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    float time_ms;
};

#endif
