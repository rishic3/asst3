#include <driver_types.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void scan_upsweep(int* result, int N, int two_d) {
    int two_dplus1 = 2 * two_d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = idx * two_dplus1;
    if (tid < N) {
        result[tid + two_dplus1 - 1] += result[tid + two_d - 1];
    }
}

__global__ void scan_downsweep(int* result, int N, int two_d) {
    int two_dplus1 = 2 * two_d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = idx * two_dplus1;
    if (tid < N) {
        int t = result[tid + two_d - 1];
        result[tid + two_d - 1] = result[tid + two_dplus1 - 1];
        result[tid + two_dplus1 - 1] += t;
    }
}

__global__ void excl_scan_block(int* result, int N) {
    size_t tid = threadIdx.x;
    extern __shared__ int smem[];
    
    // read (two elems per thread)
    smem[2 * tid] = result[2 * tid];
    smem[2 * tid + 1] = result[2 * tid + 1];

    // upsweep
    for (int two_d = 1; two_d < N; two_d <<= 1) {  // 1, 2, 4, 8,...
        __syncthreads();
        int two_dplus1 = two_d << 1;
        if (tid < N / two_dplus1) {  // 2, 4, 8, 16,...
            // t0: smem[0+2-1 = 1] += smem[0+1-1 = 0]
            // t1: smem[2+2-1 = 3] += smem[2+1-1 = 2]
            //
            // t0: smem[0+4-1 = 3] += smem[0+2-1 = 1]
            // t1: smem[4+4-1 = 7] += smem[4+2-1 = 5]
            //
            // t0: smem[0+8-1 = 7] += smem[0+4-1 = 3]
            // t1: smem[8+8-1 = 15] += smem[8+4-1 = 11]
            smem[tid * two_dplus1 + two_dplus1 - 1] += smem[tid * two_dplus1 + two_d - 1];
        }
    }

    __syncthreads();
    if (tid == 0) smem[N - 1] = 0;

    // downsweep
    for (int two_d = N >> 1; two_d > 0; two_d >>= 1) {
        __syncthreads();
        int two_dplus1 = two_d << 1;
        if (tid < N / two_dplus1) {
            int t = smem[tid * two_dplus1 + two_d - 1];
            smem[tid * two_dplus1 + two_d - 1] = smem[tid * two_dplus1 + two_dplus1 - 1];
            smem[tid * two_dplus1 + two_dplus1 - 1] += t;
        }
    }

    // write (two elems per thread)
    __syncthreads();
    result[2 * tid] = smem[2 * tid];
    result[2 * tid + 1] = smem[2 * tid + 1];
}


__global__ void excl_scan(int* result, int N, int* sums) {
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 2;
    extern __shared__ int smem[];
    
    // read (two elems per thread)
    smem[2 * tid] = result[block_offset + 2 * tid];
    smem[2 * tid + 1] = result[block_offset + 2 * tid + 1];

    // upsweep
    for (int two_d = 1; two_d < N; two_d <<= 1) {
        __syncthreads();
        int two_dplus1 = two_d << 1;
        if (tid < N / two_dplus1) {
            smem[tid * two_dplus1 + two_dplus1 - 1] += smem[tid * two_dplus1 + two_d - 1];
        }
    }

    // write scan sum to block_sums
    __syncthreads();
    if (tid == 0) {
        sums[blockIdx.x] = smem[N - 1];
        smem[N - 1] = 0;
    }

    // downsweep
    for (int two_d = N >> 1; two_d > 0; two_d >>= 1) {
        __syncthreads();
        int two_dplus1 = two_d << 1;
        if (tid < N / two_dplus1) {
            int t = smem[tid * two_dplus1 + two_d - 1];
            smem[tid * two_dplus1 + two_d - 1] = smem[tid * two_dplus1 + two_dplus1 - 1];
            smem[tid * two_dplus1 + two_dplus1 - 1] += t;
        }
    }

    // write (two elems per thread)
    __syncthreads();
    result[block_offset + 2 * tid] = smem[2 * tid];
    result[block_offset + 2 * tid + 1] = smem[2 * tid + 1];
}

__global__ void uniform_add(int* result, int* incr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] += incr[blockIdx.x];
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    bool do_better = true;

    if (!do_better) {  
        // original upsweep, downsweep implementation with kernel at every iteration.
        N = nextPow2(N);
        for (int two_d = 1; two_d <= N/2; two_d *= 2) {
            int two_dplus1 = 2 * two_d;
            // for simplicity, always invoke THREADS_PER_BLOCK threads. 
            // number of blocks needed goes down as we reduce.
            int n_active_threads = N / two_dplus1;
            int n_blocks = (n_active_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            scan_upsweep<<<n_blocks, THREADS_PER_BLOCK>>>(result, N, two_d); 
        }

        cudaMemset(&result[N-1], 0, sizeof(int));

        for (int two_d = N/2; two_d >= 1; two_d /= 2) {
            int two_dplus1 = 2 * two_d;
            int n_active_threads = N / two_dplus1;
            int n_blocks = (n_active_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            scan_downsweep<<<n_blocks, THREADS_PER_BLOCK>>>(result, N, two_d); 
        }
    } else {
        // better version with single kernel.
        N = nextPow2(N);
        const int ELEMS_PER_BLOCK = (THREADS_PER_BLOCK * 2);

        if (N <= ELEMS_PER_BLOCK) {
            // run single-block exclusive scan if the array fits in a block
            excl_scan_block<<<1, N / 2, N * sizeof(int)>>>(result, N);
            return;
        }
        
        const int n_blocks = N / ELEMS_PER_BLOCK;
        const int smem_size = ELEMS_PER_BLOCK * sizeof(int);
        
        int* sums;
        int* incr;
        cudaMalloc(&sums, n_blocks * sizeof(int));
        cudaMalloc(&incr, n_blocks * sizeof(int));

        // run multi-block excl scan
        excl_scan<<<n_blocks, THREADS_PER_BLOCK, smem_size>>>(result, ELEMS_PER_BLOCK, sums);
        cudaMemcpy(incr, sums, n_blocks * sizeof(int), cudaMemcpyDeviceToDevice);
        
        // recurse on sums and store in incr
        exclusive_scan(sums, n_blocks, incr);

        // add incr[j] to all elements in block j
        uniform_add<<<n_blocks, ELEMS_PER_BLOCK>>>(result, incr);

        cudaFree(sums);
        cudaFree(incr);
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;

    printf("inarray:     [");
    for (int i = 0; i  < 10; ++i) {
        printf(" %d ", inarray[i]);
    }
    printf("]\n");

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    printf("resultarray: [");
    for (int i = 0; i  < 10; ++i) {
        printf(" %d ", resultarray[i]);
    }
    printf("]\n");

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__ void flag_repeats(int* input, int length, int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length - 1) {
        output[idx] = (input[idx] == input[idx+1]) ? 1 : 0;
    }
}

__global__ void scatter(int* scan, int* flags, int length, int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        if (flags[idx]) {
            output[scan[idx]] = idx;
        }
    }
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    int rounded_length = nextPow2(length);
    int num_blocks = (rounded_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // get flags where repeats occur
    int* flags;
    cudaMalloc(&flags, rounded_length * sizeof(int));
    cudaMemset(flags, 0, rounded_length * sizeof(int));
    flag_repeats<<<num_blocks, THREADS_PER_BLOCK>>>(device_input, length, flags);

    // excl scan to get indices in output array
    int* scan;
    cudaMalloc(&scan, rounded_length * sizeof(int));
    cudaMemcpy(scan, flags, rounded_length * sizeof(int), cudaMemcpyDeviceToDevice);
    exclusive_scan(flags, length, scan);

    // length is end of scan
    // (to be precise it is scan[length-1] + flags[length-1], but flags[length-1] is always zero)
    int num_pairs;
    cudaMemcpy(&num_pairs, &scan[length - 1], sizeof(int), cudaMemcpyDeviceToHost);

    // scatter into output
    scatter<<<num_blocks, THREADS_PER_BLOCK>>>(scan, flags, length, device_output);

    cudaFree(scan);
    cudaFree(flags);

    return num_pairs; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
