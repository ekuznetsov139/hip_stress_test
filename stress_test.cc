#include <hip/hip_runtime.h>
#include <unistd.h>

#include <vector>
#include <chrono>
#include <atomic>
__global__ void null_kernel(const uint8_t* ptr)
{
  __shared__ int temp[256];
  temp[threadIdx.x] = sinf(float(threadIdx.x));
}

void rocm_null_gpu_job(void* stream, const uint8_t* ptr)
{
  hipLaunchKernelGGL(null_kernel,
                     1,
                     256, 0, (hipStream_t)stream, ptr);
}

std::vector<std::vector<hipStream_t> > stream_pool;

std::atomic<int> counter(0);
bool do_kill = false;

std::chrono::system_clock::time_point thread_reports[16];

void thread_job(int dev, int virt) {
    hipSetDevice(dev);
    uint8_t* mem;
    hipMalloc(&mem, 512);
    uint8_t hmem[512];

    hipStream_t exec_stream = stream_pool[dev][virt];
    hipStream_t h2d_stream = stream_pool[dev][virt+4];
    hipStream_t d2h_stream = stream_pool[dev][virt+8];
    hipEvent_t eh2d, ed2h;
    hipEventCreate(&eh2d);
    hipEventCreate(&ed2h);  
    uint64_t n=0;

    while(!do_kill) {
        rocm_null_gpu_job(exec_stream, mem);
        hipMemcpyAsync(hmem, mem, 4, hipMemcpyDeviceToHost, d2h_stream);
        hipMemcpyAsync(mem+256, hmem+256, 4, hipMemcpyHostToDevice, h2d_stream);
        hipEventRecord(eh2d, h2d_stream);
        hipEventRecord(ed2h, d2h_stream);
        //hipStreamWaitEvent(exec_stream, eh2d, 0);
        //hipStreamWaitEvent(exec_stream, ed2h, 0);
        n++;
        if ((n&15) == 0) {
            hipStreamSynchronize(exec_stream);
            hipStreamSynchronize(h2d_stream);
            hipStreamSynchronize(d2h_stream);
            thread_reports[dev*4+virt] = std::chrono::system_clock::now();
        }
        counter++;
    }
    hipStreamSynchronize(exec_stream);
    hipStreamSynchronize(h2d_stream);
    hipStreamSynchronize(d2h_stream);

    hipFree(mem);
    hipEventDestroy(eh2d);
    hipEventDestroy(ed2h);
}

int main()
{
    stream_pool.resize(4);
    std::vector<uint8_t*> memory_buffers[4];
    for(int i=0; i<4; i++) {
        hipSetDevice(i);
        stream_pool[i].resize(12);
        memory_buffers[i].resize(128);
        for(int j=0; j<12; j++)
            hipStreamCreate(&stream_pool[i][j]);
        for(int j=0; j<128; j++)
            hipMalloc(&memory_buffers[i][j], 4096 * ((j&1)+1));
    }

    for(int nDev=1; nDev<=4; nDev++) {
        counter = 0;
        printf("RUNNING ON %d DEVICES\n", nDev);
        do_kill = false;
        std::vector<std::thread> threads;
        for(int i=0; i<nDev*4; i++)
            threads.push_back(std::thread(thread_job, i/4, i%4));
        usleep(1000000);
        auto t1 = std::chrono::system_clock::now();
        int count = int(counter);
        uint64_t total_count = 0;
        double total_time = 0;
        for (int t=0; t<30; t++) {
            usleep(1000000);
            auto t2 = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            int count2 = int(counter);
            printf("%.3f ms .... %d (%.3f job/s)\n", duration/1000.0, count2-count, ((count2-count)*1000000.)/duration);
            for(int i=0; i<nDev*4; i++) {
                if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - thread_reports[i]).count() >= 1000000) {
                    printf("Thread %d/%d is stuck\n", i/4, i%4);
                }
            }
            total_count += count2-count;
            total_time += duration*1e-6;
            t1 = t2;
            count = count2;
        }
        printf("AVERAGE: %ld / %f = %f job/s\n", total_count, total_time, total_count/total_time);
        printf("Shutting down...\n");
        do_kill = true;
        t1 = std::chrono::system_clock::now();
        for(auto& t: threads)
            t.join();
        auto t2 = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        printf("Shut down complete in %d us\n", duration);
        for (int i=0; i<nDev; i++) {
            hipSetDevice(i);
            hipDeviceSynchronize();
        }
    }
}
