# Part 1: SAXPY

## Question 1

**What performance do you observe compared to the sequential CPU-based implementation of SAXPY (recall your results from saxpy on Program 5 from Assignment 1)?**

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

## Question 2

**Compare and explain the difference between the results provided by two sets of timers (timing only the kernel execution vs. timing the entire process of moving data to the GPU and back in addition to the kernel execution). Are the bandwidth values observed roughly consistent with the reported bandwidths available to the different components of the machine? (You should use the web to track down the memory bandwidth of an NVIDIA T4 GPU. Hint: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf. The expected bandwidth of memory bus of AWS is 5.3 GB/s, which does not match that of a 16-lane PCIe 3.0. Several factors prevent peak bandwidth, including CPU motherboard chipset performance and whether or not the host CPU memory used as the source of the transfer is “pinned” — the latter allows the GPU to directly access memory without going through virtual memory address translation. If you are interested, you can find more info here: https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory)**

# Part 2: Exclusive Scan and Find Repeats

### Initial implementation:

Scan tests
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

Find repeats tests
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

### Do better enabled:

Scan tests
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

Find repeats tests
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

# Part 3: Circle Render

<b>We would like you to hand in a clear, high-level description of how your implementation works as well as a brief description of how you arrived at this solution. Specifically address approaches you tried along the way, and how you went about determining how to optimize your code (For example, what measurements did you perform to guide your optimization efforts?).

Aspects of your work that you should mention in the write-up include:

1. Include both partners names and SUNet id's at the top of your write-up.
2. Replicate the score table generated for your solution and specify which machine you ran your code on.
3. Describe how you decomposed the problem and how you assigned work to CUDA thread blocks and threads (and maybe even warps).
4. Describe where synchronization occurs in your solution.
5. What, if any, steps did you take to reduce communication requirements (e.g., synchronization or main memory bandwidth requirements)?
6. Briefly describe how you arrived at your final solution. What other approaches did you try along the way. What was wrong with them?</b>

So our two core initial issues are atomicity and ordering. Currently, each thread in the `kernelRenderCircles` kernel is assigned a circle, and they all run concurrently. Thus there is no atomicity guarantee when reading and writing to potentially overlapping pixels, nor is there an ordering guarantee when processing circles.

My immediate thinking is that logically, we should group circles that overlap together, and separate groups that do not overlap. Within each group, the circles should be ordered by input order and processed sequentially. Visually, this would look a bit like a histogram where buckets are ordered by input order.

The next question was how to parallelize this. The naive first thing I thought of was assigning each group of circles to a logical thread, and then each thread processes its group of circles sequentially. But this doesn't really map well to CUDA hardware, and if we did this literally we wouldn't be leveraging the second dimension of parallelism (parallelism across pixels). Compute-wise the pixel grid is ideal for a 1:1 thread-to-pixel mapping.

## first approach

So my initial approach was to have each thread process a single pixel, and have all threads walk through **all circles** and apply any relevant updates to that pixel. This is obviously not efficient from a work perspective (checking every circle against every pixel) or a coherence perspective, and I'm noting the hint about using `circleBoxTest.cu_inl`. But in the spirit of doing the easiest thing first, this leads us to a very simple kernel work assignment, and solves the atomicity issue (which obviates the ordering issue as well).

My initial implementation:
```text
for each pixel (parallelized across threads)
    get my (x,y) coordinates
    read current pixel value
    for each circle (sequentially)
      if my pixel is within the circle
        blend contribution of circle into image for my pixel
    write my final accumulated color to global memory
```

With the following kernel:
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

The first pass resolved the correctness issues, which was a good sign for the first pass. But the performance was poor, as expected given the obvious inefficencies we decided to accept for the initial approach.
```text
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.0052           | 0.089           | 2               |
| rand10k         | 0.0048           | 26.4764         | 2               |
| rand100k        | 0.0046           | 259.8125        | 2               |
| pattern         | 0.0059           | 3.247           | 2               |
| snowsingle      | 0.0046           | 140.7397        | 2               |
| biglittle       | 0.0045           | 26.5156         | 2               |
| rand1M          | 0.0048           | 2541.1458       | 2               |
| micro2M         | 0.0043           | 5215.9366       | 2               |
--------------------------------------------------------------------------
|                                    | Total score:    | 16/72           |
--------------------------------------------------------------------------
```

## second approach

The next step is fairly clear; we want to avoid the inefficient pixels x circles comparisons. The provided `circleBoxTest.cu_inl` is an implication that we can do some pre-filtering to try to restrict the groups that any given pixel (thread) needs to look at when checking whether that circle contributes. 

At the moment, the filtering is done at a 'per-pixel granularity', so to speak. What we could do instead is increase the granularity to each block. Each block can do an initial filtration — e.g., the block cooperatively filters the list into just the overlapping circles for the given block — and then proceeds into the same thread-per-pixel kernel, but now each pixels has a much smaller sub-list in its consideration set.

My idea here is to compute the bounding box of the block, and in the prefiltering stage, each thread will handle a subset of the circles and check the overlap with the block. Any overlapping circles will be added to a candidate list in shared memory. As a result, we only need to iterate over the (hopefully much smaller) candidate list in the shading loop.

Below, I am anticipating that the candidate circles could exceed the shared memory size in rare cases. As a naive fallback, I've added an overflow flag that will fallback to the slower loop that iterates over all circles.

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

After testing my implementation:

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
...one of the correctness criteria was broken. Notably each thread is atomically updating an index to track the number of candidates, but this doesn't preserve the *ordering* of the candidates, now that we're processing circles in parallel.

