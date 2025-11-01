# Assignment 3 Writeup

**SUNET IDS**: `rishic3`, `madani49`

## Part 1: SAXPY

**Results:**

```text
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA T4G
   SMs:        40
   Global mem: 14914 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Running 3 timing tests:
Effective BW by CUDA saxpy: 201.533 ms          [5.545 GB/s]
Kernel time by CUDA saxpy: 17.693 ms
Effective BW by CUDA saxpy: 207.894 ms          [5.376 GB/s]
Kernel time by CUDA saxpy: 4.862 ms
Effective BW by CUDA saxpy: 207.892 ms          [5.376 GB/s]
Kernel time by CUDA saxpy: 4.863 ms
```

### Question 1

In CPU sequential SAXPY we observed roughly 10ms performance when parallelizing over ISPC on CPU cores. Here with far more cores we achieve only ~2x speedup in our best run. This can be attributed to the fact that firstly — as we observed in assignment 1 — our program is memory bound, and additionally the working set is likely not large enough to hide the memory transfer latency between host CPU and device GPU memory. All of these factors work against our speedup, despite the massive amount of parallel compute available.

### Question 2

No, the BW does not match the theoretical maximums on the datasheet for either 300GB/s DRAM or 32GB/s PCIE links. This is explained by the fact that memory transfer time for this program is dominated by the memory copies to and from the Host CPU, which means that in reality, we are predictably limited by the lower throughput AWS memory bus (rated 5.3GB/s). This aligns with the effective BW numbers we observe above.

## Part 2: Exclusive Scan and Find Repeats

We made two implementations, shown in `exclusive_scan()` with and without the `do_better` flag. The initial implementation (using kernels `scan_upsweep` and `scan_downsweep`) follows the README directions, with an upsweep and downsweep kernel, which are iteratively invoked with decreasing/increasing granularity respectively. Using this implementation, we see the following numbers:

```text
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.638           | 0.479           | 1.25            |
| 10000000        | 8.925           | 7.817           | 1.25            |
| 20000000        | 17.659          | 15.601          | 1.25            |
| 40000000        | 35.216          | 30.972          | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5.0/5.0         |
-------------------------------------------------------------------------
```

```text
-------------------------
Find_repeats Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 1.037           | 0.817           | 1.25            |
| 10000000        | 11.966          | 10.622          | 1.25            |
| 20000000        | 21.418          | 20.081          | 1.25            |
| 40000000        | 42.471          | 39.054          | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5.0/5.0         |
-------------------------------------------------------------------------
```

We felt we could do better by avoiding the many (2logN) kernel launches and repeated reads/writes to global memory by combining the sweeps into a single kernel, and performing the scan in shared memory. This implementation was actually what we felt to be more intuitive from the start of this problem. 

The improved implementation (using kernels `excl_scan_block`, `excl_scan`, and `uniform_add`) uses a recursive approach to perform the scan in shared memory (the upsweep/downsweep loops are brought into the kernel), collect the sums into an intermediate array, and then recursively scan the block-level sums. This continues until the working set is small enough to fit in a single block, at which point we run a final block-level kernel. The scanned sums can be added back uniformly to the block's elements using a simple kernel. We observed the following improvements (which are notable—up to ~2.5x improvement over the initial implementation—at large scales):

```text
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.642           | 0.469           | 1.25            |
| 10000000        | 8.874           | 2.371           | 1.25            |
| 20000000        | 17.678          | 4.461           | 1.25            |
| 40000000        | 35.214          | 8.639           | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5.0/5.0         |
-------------------------------------------------------------------------
```

```text
-------------------------
Find_repeats Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 1.066           | 0.811           | 1.25            |
| 10000000        | 11.944          | 5.175           | 1.25            |
| 20000000        | 21.427          | 8.911           | 1.25            |
| 40000000        | 41.319          | 16.404          | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5.0/5.0         |
-------------------------------------------------------------------------
```

## Part 3: Circle Render

<b>We would like you to hand in a clear, high-level description of how your implementation works as well as a brief description of how you arrived at this solution. Specifically address approaches you tried along the way, and how you went about determining how to optimize your code (For example, what measurements did you perform to guide your optimization efforts?).

Aspects of your work that you should mention in the write-up include:

1. Include both partners names and SUNet id's at the top of your write-up.
2. Replicate the score table generated for your solution and specify which machine you ran your code on.
3. Describe how you decomposed the problem and how you assigned work to CUDA thread blocks and threads (and maybe even warps).
4. Describe where synchronization occurs in your solution.
5. What, if any, steps did you take to reduce communication requirements (e.g., synchronization or main memory bandwidth requirements)?
6. Briefly describe how you arrived at your final solution. What other approaches did you try along the way. What was wrong with them?</b>

Our two core initial issues are atomicity and ordering. In the starter code, each thread in the `kernelRenderCircles` kernel is assigned a circle, and they all run concurrently, handling potentially overlapping pixels. Thus there is no atomicity guarantee when two threads are reading and writing to the same pixels, nor is there an ordering guarantee when processing circles.

Our first thought was that logically, we should group circles that overlap together, and separate groups that do not overlap. Within each group, the circles should be ordered by input order and processed sequentially. Visually, we imagined something like a histogram where buckets are ordered by input order.

The next question was how to parallelize this. The naive first thing we thought of was assigning each group of circles to a logical thread, and then each thread processes its group of circles sequentially. But this doesn't really map well to CUDA hardware, and if we did this literally we wouldn't be leveraging the second dimension of parallelism mentioned in the README: parallelism across pixels. Compute-wise, it makes sense for pixels to be mapped 1:1 to threads, as this simplifies how the image is split up.

### First Approach

This led us to our first approach, which was to have each thread process a single pixel, and have all threads walk through **all circles** and apply any relevant updates to that pixel. This is obviously not efficient from a work perspective (checking every circle against every pixel) or a coherence perspective, and we noted the hint about using `circleBoxTest.cu_inl`. But in the spirit of doing the easiest thing first, this lead us to a very simple static work assignment across pixels, and it solved the atomicity issue (and in doing so, it obviated the ordering issue as well, since each thread walks through circles in order). No synchronization was required, since pixels are handled entirely independently.

Our pseudo-code:
```text
for each pixel (parallelized across threads)
    get my (x,y) coordinates
    read current pixel value
    for each circle (sequentially)
      if my pixel is within the circle
        blend contribution of circle into image for my pixel
    write my final accumulated color to global memory
```

And the kernel:
```cpp
template<SceneName scene>
__global__ void kernelRenderCircles() {
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    // bounds check
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;

    // get my normalized pixel coordinates
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float pixelCenterNormX = invWidth * (static_cast<float>(pixelX) + 0.5f);
    float pixelCenterNormY = invHeight * (static_cast<float>(pixelY) + 0.5f);

    // global memory read of pixel (independent per thread)
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float4 newColor = *imgPtr;

    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    // iterate over circles sequentially
    for (int circleIndex = 0; circleIndex < cuConstRendererParams.numCircles; ++circleIndex) {
        // get circle
        int index3 = 3 * circleIndex;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);

        // --- in-lined logic for shadePixel ---
        float rad = cuConstRendererParams.radius[circleIndex];
        float maxDist = rad * rad;
        float diffX = p.x - pixelCenterNormX;
        float diffY = p.y - pixelCenterNormY;
        float pixelDist = diffX * diffX + diffY * diffY;

        // circle does not contribute to this pixel
        if (pixelDist > maxDist) continue;

        float3 rgb;
        float alpha;

        if constexpr (scene == SNOWFLAKES || scene == SNOWFLAKES_SINGLE_FRAME) {
            float normPixelDist = sqrt(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);

            float maxAlpha = .6f + .4f * (1.f - p.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
        } else {
            rgb = *(float3*)(&cuConstRendererParams.color[index3]);
            alpha = .5f;
        }

        float oneMinusAlpha = 1.f - alpha;

        // update new color with this circle's contribution
        newColor.x = alpha * rgb.x + oneMinusAlpha * newColor.x;
        newColor.y = alpha * rgb.y + oneMinusAlpha * newColor.y;
        newColor.z = alpha * rgb.z + oneMinusAlpha * newColor.z;
        newColor.w += alpha;
    }

    *imgPtr = newColor;
}
```

We observed the following results on the AWS T4G machine. We were happy to see the correctness issues resolved, but as we expected, the performance was poor given the inefficencies we decided to accept for the initial approach.

```text
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2502           | 0.2459          | 9               |
| rand10k         | 3.0614           | 69.7325         | 2               |
| rand100k        | 29.6683          | 721.9569        | 2               |
| pattern         | 0.3933           | 7.9576          | 2               |
| snowsingle      | 19.6507          | 431.9749        | 2               |
| biglittle       | 15.2659          | 72.233          | 4               |
| rand1M          | 235.7366         | 7899.5876       | 2               |
| micro2M         | 453.8863         | 18524.8016      | 2               |
--------------------------------------------------------------------------
|                                    | Total score:    | 25/72           |
--------------------------------------------------------------------------
```

### Second Approach

Given the helpful hints in the README, our next step was fairly clear: we wanted to avoid the inefficient `pixels x circles` comparisons, and we wanted parallelism across circles. Some threads are checking circles that are nowhere near their pixel, so we could be smarter about which circles to check. The provided `circleBoxTest.cu_inl` is an implication that we can do some pre-filtering to try to restrict the groups that any given thread needs to look at when checking whether that circle contributes to their pixel.

Our high-level thought was that, since we have a nice static assignment of thread blocks to pixel blocks, each block can do an initial filtration — e.g., the block cooperatively filters the list into just the overlapping circles for the given block — and then proceeds into the same thread-per-pixel kernel, but now each pixel has a (hopefully much smaller) sub-list in its consideration set when doing the shading.

For cooperation, each thread would handle a subset of the circles and check the overlap with the block. Any overlapping circles would be added to a candidate list in shared memory. 

While implementing this, we realized that the number of candidate circles could exceed shared memory size. As a simple fallback, we added an overflow flag that would force the kernel to fall back to our slower first approach that iterates over all circles.

```cpp
template<SceneName scene>
__global__ void kernelRenderCircles() {
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // --- pre-filtration ---
    __shared__ int candidate_circles[MAX_CIRCLES_PER_BLOCK];
    __shared__ int num_candidates;
    __shared__ bool overflow;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        num_candidates = 0;
        overflow = false;
    }
    __syncthreads();

    // compute bounding box of our block
    float blockMinX = blockIdx.x * blockDim.x * invWidth;
    float blockMaxX = (blockIdx.x + 1) * blockDim.x * invWidth;
    float blockMinY = blockIdx.y * blockDim.y * invHeight;
    float blockMaxY = (blockIdx.y + 1) * blockDim.y * invHeight;

    // check circle overlap in parallel
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < cuConstRendererParams.numCircles; i += blockDim.x * blockDim.y) {
        int index3 = 3 * i;
        float rad = cuConstRendererParams.radius[i];
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);

        // use true circle in box test, the candidate list is guaranteed to overlap with the block (but not necessarily the pixel).
        if (circleInBox(p.x, p.y, rad, blockMinX, blockMaxX, blockMaxY, blockMinY)) {
            int index = atomicAdd(&num_candidates, 1);
            if (index >= MAX_CIRCLES_PER_BLOCK) {
                // exceeded smem size, break
                overflow = true;
            } else {
                candidate_circles[index] = i;
            }
        }
    }
    __syncthreads();

    // --- pixel processing ---
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;

    // get my normalized pixel coordinates
    float pixelCenterNormX = invWidth * (static_cast<float>(pixelX) + 0.5f);
    float pixelCenterNormY = invHeight * (static_cast<float>(pixelY) + 0.5f);

    // global memory read of pixel (independent per thread)
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float4 newColor = *imgPtr;

    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    // overflow case - fallback to considering all circles as candidates
    if (overflow) num_candidates = cuConstRendererParams.numCircles;

    // iterate over candidate circles sequentially
    for (int i = 0; i < num_candidates; ++i) {
        int circleIndex = (overflow) ? i : candidate_circles[i];
        int index3 = 3 * circleIndex;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);

        // --- shadePixel ---
        float rad = cuConstRendererParams.radius[circleIndex];
        float maxDist = rad * rad;
        float diffX = p.x - pixelCenterNormX;
        float diffY = p.y - pixelCenterNormY;
        float pixelDist = diffX * diffX + diffY * diffY;

        // circle does not contribute to this pixel
        if (pixelDist > maxDist) continue;

        float3 rgb;
        float alpha;

        if constexpr (scene == SNOWFLAKES || scene == SNOWFLAKES_SINGLE_FRAME) {
            float normPixelDist = sqrt(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);

            float maxAlpha = .6f + .4f * (1.f - p.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
        } else {
            rgb = *(float3*)(&cuConstRendererParams.color[index3]);
            alpha = .5f;
        }

        float oneMinusAlpha = 1.f - alpha;

        // update new color with this circle's contribution
        newColor.x = alpha * rgb.x + oneMinusAlpha * newColor.x;
        newColor.y = alpha * rgb.y + oneMinusAlpha * newColor.y;
        newColor.z = alpha * rgb.z + oneMinusAlpha * newColor.z;
        newColor.w += alpha;
    }

    *imgPtr = newColor;
}
```

After testing our implementation:

```
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2524           | 0.2494          | 9               |
| rand10k         | 3.0647           | (F)             | 0               |
| rand100k        | 29.5738          | (F)             | 0               |
| pattern         | 0.3938           | (F)             | 0               |
| snowsingle      | 19.6292          | 4.7903          | 9               |
| biglittle       | 15.1614          | (F)             | 0               |
| rand1M          | 226.4349         | (F)             | 0               |
| micro2M         | 434.9885         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 18/72           |
--------------------------------------------------------------------------
```
...we realized we broke our correctness criteria. Each thread was atomically updating an index to track the number of candidates, but this was not preserving the *ordering* of the candidates as they are appended to the list, now that we're processing circles in parallel.

A new challenge was to get the candidate circles back in sorted order. Using a hint from the README (and on EdStem - thanks Weixin!), we realized we could do this using exclusive scan, similar to part 2 of the assignment, as shown below:

```cpp
template<SceneName scene>
__global__ void kernelRenderCircles() {
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // --- pre-filtration ---
    __shared__ int candidate_circles[MAX_CIRCLES_PER_BLOCK];
    __shared__ int num_candidates;
    __shared__ bool overflow;

    __shared__ uint prefix_sum_input[SCAN_BLOCK_DIM];
    __shared__ uint prefix_sum_output[SCAN_BLOCK_DIM];
    __shared__ uint prefix_sum_scratch[2 * SCAN_BLOCK_DIM];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        num_candidates = 0;
        overflow = false;
    }
    __syncthreads();

    // compute bounding box of our block
    float blockMinX = blockIdx.x * blockDim.x * invWidth;
    float blockMaxX = (blockIdx.x + 1) * blockDim.x * invWidth;
    float blockMinY = blockIdx.y * blockDim.y * invHeight;
    float blockMaxY = (blockIdx.y + 1) * blockDim.y * invHeight;

    int linearThreadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int numCircles = cuConstRendererParams.numCircles;

    // check circle overlap in parallel
    for (int blk = 0; blk < numCircles; blk += SCAN_BLOCK_DIM) {
        int circleIndex = blk + linearThreadIndex;
        int is_in_box = 0;

        // each thread checks a single circle in block
        if (circleIndex < numCircles) {
            int index3 = 3 * circleIndex;
            float rad = cuConstRendererParams.radius[circleIndex];
            float3 p = *(float3*)(&cuConstRendererParams.position[index3]);

            // use true circle in box test
            is_in_box = circleInBox(p.x, p.y, rad, blockMinX, blockMaxX, blockMaxY, blockMinY);
        }

        // run scan to turn is_in_box flags into write locations
        prefix_sum_input[linearThreadIndex] = is_in_box;
        __syncthreads();
        sharedMemExclusiveScan(linearThreadIndex, prefix_sum_input, prefix_sum_output, prefix_sum_scratch, SCAN_BLOCK_DIM);
        __syncthreads();

        // if is in box and max circles not yet exceeded, write to candidates
        // note write loc is offset by num_candidates to account for block
        uint write_loc_offset = prefix_sum_output[linearThreadIndex];
        if (is_in_box && (num_candidates + write_loc_offset < MAX_CIRCLES_PER_BLOCK)) {
            candidate_circles[num_candidates + write_loc_offset] = circleIndex;
        }
        __syncthreads();

        // last thread updates total count added in this block
        if (linearThreadIndex == SCAN_BLOCK_DIM - 1) {
            int new_num_candidates = num_candidates + write_loc_offset + is_in_box;
            // check if we've exceeded capacity, in which case set overflow
            if (new_num_candidates > MAX_CIRCLES_PER_BLOCK) overflow = true;
            num_candidates = min(new_num_candidates, MAX_CIRCLES_PER_BLOCK);
        }
        __syncthreads();

        if (num_candidates >= MAX_CIRCLES_PER_BLOCK) break;
    }
    __syncthreads();

    // --- pixel processing ---
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;

    // get my normalized pixel coordinates
    float pixelCenterNormX = invWidth * (static_cast<float>(pixelX) + 0.5f);
    float pixelCenterNormY = invHeight * (static_cast<float>(pixelY) + 0.5f);

    // global memory read of pixel (independent per thread)
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float4 newColor = *imgPtr;

    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    // if overflow, we'll check all circles as a fallback
    int candidates_to_check = overflow ? numCircles : num_candidates;

    // iterate over candidate circles sequentially
    for (int i = 0; i < candidates_to_check; ++i) {
        // no overflow: index from candidates, overflow: index entire range
        int circleIndex = overflow ? i : candidate_circles[i];
        int index3 = 3 * circleIndex;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);

        // --- shadePixel ---
        float rad = cuConstRendererParams.radius[circleIndex];
        float maxDist = rad * rad;
        float diffX = p.x - pixelCenterNormX;
        float diffY = p.y - pixelCenterNormY;
        float pixelDist = diffX * diffX + diffY * diffY;

        // circle does not contribute to this pixel
        if (pixelDist > maxDist) continue;

        float3 rgb;
        float alpha;

        if constexpr (scene == SNOWFLAKES || scene == SNOWFLAKES_SINGLE_FRAME) {
            float normPixelDist = sqrt(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);

            float maxAlpha = .6f + .4f * (1.f - p.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
        } else {
            rgb = *(float3*)(&cuConstRendererParams.color[index3]);
            alpha = .5f;
        }

        float oneMinusAlpha = 1.f - alpha;

        // update new color with this circle's contribution
        newColor.x = alpha * rgb.x + oneMinusAlpha * newColor.x;
        newColor.y = alpha * rgb.y + oneMinusAlpha * newColor.y;
        newColor.z = alpha * rgb.z + oneMinusAlpha * newColor.z;
        newColor.w += alpha;
    }

    *imgPtr = newColor;
}
```

With this, we were able to restore correctness and meet the performance requirements. Hooray!
```text
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2534           | 0.254           | 9               |
| rand10k         | 3.0598           | 2.963           | 9               |
| rand100k        | 29.667           | 28.2294         | 9               |
| pattern         | 0.3951           | 0.4078          | 9               |
| snowsingle      | 19.6878          | 19.8177         | 9               |
| biglittle       | 15.1899          | 14.1093         | 9               |
| rand1M          | 232.4888         | 233.6674        | 9               |
| micro2M         | 448.0229         | 450.0085        | 9               |
--------------------------------------------------------------------------
|                                    | Total score:    | 72/72           |
--------------------------------------------------------------------------
```

### Optimizations

To look for optimization opportunities within the kernel, we profiled rand1M with NCU, using this command:

```shell
$ sudo /usr/local/cuda-12.8/bin/ncu --set full ./render -c rand1M -s 1024
```

What stuck out to us was the following:

```text
    OPT   Est. Speedup: 35.04%                                                                                          
          On average, each warp of this workload spends 10.9 cycles being stalled waiting for sibling warps at a CTA    
          barrier. A high number of warps waiting at a barrier is commonly caused by diverging code paths before a      
          barrier. This causes some warps to wait a long time until other warps reach the synchronization point.        
          Whenever possible, try to divide up the work into blocks of uniform workloads. If the block size is 512       
          threads or greater, consider splitting it into smaller groups. This can increase eligible warps without       
          affecting occupancy, unless shared memory becomes a new occupancy limiter. Also, try to identify which        
          barrier instruction causes the most stalls, and optimize the code executed before that synchronization point  
          first. This stall type represents about 48.2% of the total average of 22.6 cycles between issuing two         
          instructions.     
```
Our immediate next thought was about how to eliminate the number of `__syncthreads()` invocations, of which there are many. The syncs are ultimately required due to sharedMemExclusiveScan— for each block, we need an initial sync after checking circle overlap and writing that to shared memory, another sync after running the scan, a third after writing to candidates, and a fourth after updating the total count, not to mention the 3 internal barriers within the scan itself. Per the profiler, we were wasting 10.9 cycles per warp on average waiting for sibling warps to arrive at barriers.

Our goal was to keep the thread-per-pixel, collaboratively pre-filter circles within a block design. Given With insight from a very helpful blog (https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/), we were looking to 

Our immediate next thought was about how to eliminate the number of `__syncthreads()` invocations, of which there are many. The syncs are ultimately required due to sharedMemExclusiveScan— for each block, we need an initial sync after checking circle overlap and writing that to shared memory, another sync after running the scan, a third after writing to candidates, and a fourth after updating the total count, not to mention the internal barriers within the scan itself. Per the profiler, we were wasting 10.9 cycles per warp on average waiting for sibling warps to arrive at barriers.

Our goal was to keep the thread-per-pixel, collaboratively pre-filter circles within a block design. We were really keen to try to do some warp-level primitives since we recognized that warps can basically share information for free without explicit synchronization, since they execute in lockstep via SIMT anyway. With insight from this very nice NVIDIA blog (https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) it was possible to remove the block-wise prefix sum entirely.

Our idea was as follows:
- the block still cooperatively checks `is_in_box`, with each thread handing a chunk
- each warp will collect the `is_in_box` results with no synchronization using warp primitives (ballot and popc)
    - as part of this, the warp computes its *write offset within the warp, aka, the lane offset*
- we run a really small exclusive scan over just the warp counts for the entire block (256/32 = 8 elements) to compute where each warp should write - combining this with lane offset, we now have the *write offset within the block*

So we basically decompose the initial exclusive scan into two levels - warp and then block - with far fewer syncs (now only 3 vs. 7 previously) since we are sharing information within the warp for free in the first stage, and with a much smaller prefix sum since the data is already aggregated per warp.

In our final optimized version, here is what we observed on the AWS T4G:
```text
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2649           | 0.2624          | 9               |
| rand10k         | 3.0795           | 1.6906          | 9               |
| rand100k        | 29.5175          | 15.4693         | 9               |
| pattern         | 0.4083           | 0.2749          | 9               |
| snowsingle      | 19.5749          | 7.3637          | 9               |
| biglittle       | 15.2305          | 13.0302         | 9               |
| rand1M          | 230.3938         | 105.8901        | 9               |
| micro2M         | 441.3776         | 198.4038        | 9               |
--------------------------------------------------------------------------
|                                    | Total score:    | 72/72           |
--------------------------------------------------------------------------
```

While some smaller tests (like rgb, biglittle) noticed marginal improvement, the tests with many circles (rand1M and micro2M) noticed the biggest speedup. This makes sense given that every iteration per circle now had less than half the block-wide synchronizations than before.
