# File details:

* `orientation_matching.cu`:
This file contains the CUDA implementation of orientation matching.
Orientation Matching = Euclidean Distance + Heap Sort This file can be used as
a standalone and integrated to any other codes, and just needed to call the
cudaComputeKNearestNeighbors function to implement CUDA orientation matching.


## Algorithm Details

All the three folders implement different versions of orientation matching.
Orientation mathcing invovles comparison between data images and reference
images. Each data image is compared with the reference images and the N nearest
reference images are identified. The comparison is done w.r.t. Euclidean
distance between the data images and reference images. The Euclidean distance
between the data images and all the reference images is computed and the
computed distances are sorted to identify the smallest distance.
```
Orientation Matching = Euclidean Distance + Sorting Algorithm
```


### Euclidean Distance

Two versions of Euclidean distances are computed; Weighted Euclidean Distane
and Non-Weighted Euclidean Distance.

1. Non-Weighted Euclidean Distance Representation (Implemented in the
`cudaOriMatchNonWeighted` folder):
```
distance = sqrt( summation( (Ai-Bi)^2 ) ) for i=1,2,...,number of dimensions;
```
where `Ai` is the `i`th dimension of `A`, and `Bi` is the `i`th dimension of `B`.

2. Weighted Euclidean Distance Representation (Implemented in the
`cudaOriMatchWeighted` and the `cudaOriMatchMultiNodeWeighted` folders):i
```
distance = sqrt( summation( Wi*(Ai-Bi)^2 ) ) for i=1,2,...,number of dimensions;
```
where `Ai` is the `i`th dimension of `A`, `Bi` is the `i`th dimension of `B`,
and `Wi` is the `i`th dimension of `W`.


### Sorting

Heap sort is used to sort the distances computed earlier. The base algorithm is
adopted from GeeksforGeeks website https://www.geeksforgeeks.org/heap-sort/.

## Code Implementation in CUDA

### Euclidean Distance

CUDA kernel is developed to compute Euclidean distance across data images and
reference images.
* Input:
    1. Data images; Size `M x D`, where `M` is the total number of data images
       and `D` is the number of dimensions in each image.
    2. Reference images; Size `N x D`, where `N` is the total number of
       reference images and `D` is the number of dimensions in each image.
* Output:
    1. Euclidean Distance; Size `M x N`, where `M` is the total number of data
       images and `N` is the total number of reference images.
All the input and output vairables are stored linearly in memory.

The Eucldiean distance is considered as a 2D matrix and are distributed across
each thread and block in CUDA thread hierarchy.  The data images are
distributed across rows, and the reference images are distributed across
columns. Blocking techniques is used where each CUDA block consisting of equal
number of threads in two-dimension. In the current implementations, each CUDA
block consists of a total of 256 CUDA threads with 16 rows and 16 columns. In a
basic implemenation, each thread computes Euclidean distance between one data
image and one reference image.  So, each CUDA block computes Euclidean distance
between a set of data images and reference images, by iteratively loading a
block of 16 dimensions of each data and reference images.

In the current implementations, each thread computes Euclidean distance between
`n` data and reference images (where `n=4` for Weighted implementation, and
`n=6` for Non-Weighted implementation). This computation of distance between
multiple data and reference images result in better utilization of GPU
resources, and improves performance.

### Optimized Euclidean Distance

The optimized euclidean distance uses the expansion of (x - y)^2 and re-formats
it into matrix operations to take advantage of Tensor Cores. 

### Sorting

The algorithm adopted from GeeksforGeeks is ported onto GPU using CUDA.

### Optimized Argmin

Optimized takes advantage of CUB Device Segmented Argmin to use tree based reductions for the GPUs.
