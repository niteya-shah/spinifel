import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from spinifel.sequential import pyCudaKNearestNeighbors_SP as pyCu_SP
from spinifel.sequential import pyCudaKNearestNeighbors_DP as pyCu_DP


def calc_eudist_gpu_sp(model_slices, slices, deviceId):
    model_slices_flat = model_slices.flatten()
    slices_flat       = slices.flatten()

    euDist = pyCu_SP.cudaEuclideanDistance(model_slices_flat,
                                           slices_flat,
                                           model_slices.shape[0],
                                           slices.shape[0],
                                           slices.shape[1],
                                           deviceId)
    return euDist

def calc_argmin_gpu_sp(euDist, n_images, n_refs, n_pixels, deviceId):

    index =  pyCu_SP.cudaHeapSort(euDist,
                               n_images,
                               n_refs,
                               n_pixels,
                               1,
                               deviceId)
    return index

def calc_eudist_gpu_dp(model_slices, slices, deviceId):
    model_slices_flat = model_slices.flatten()
    slices_flat       = slices.flatten()

    euDist = pyCu_DP.cudaEuclideanDistance(model_slices_flat,
                                           slices_flat,
                                           model_slices.shape[0],
                                           slices.shape[0],
                                           slices.shape[1],
                                           deviceId)
    return euDist

def calc_argmin_gpu_dp(euDist, n_images, n_refs, n_pixels, deviceId):

    index =  pyCu_DP.cudaHeapSort(euDist,
                               n_images,
                               n_refs,
                               n_pixels,
                               1,
                               deviceId)
    return index

def compare_cpu_gpu_knn():
    detectorSize = 128*128
    num_models = 512
    num_slices = 1024
    model_slices = np.random.rand(num_models, detectorSize)
    slices = np.random.rand(num_slices, detectorSize)
    deviceId = 0

    cpu_eu_dist = euclidean_distances(model_slices, slices, squared=True)
    gpu_eu_dist_sp = calc_eudist_gpu_sp(model_slices, slices, deviceId)
    gpu_eu_dist_dp = calc_eudist_gpu_dp(model_slices, slices, deviceId)

    cpu_indices = np.argmin(cpu_eu_dist, axis=0)
    gpu_indices_sp = calc_argmin_gpu_sp(gpu_eu_dist_sp,
                                        slices.shape[0],
                                        model_slices.shape[0],
                                        slices.shape[1],
                                        deviceId)
    gpu_indices_dp = calc_argmin_gpu_dp(gpu_eu_dist_dp,
                                        slices.shape[0],
                                        model_slices.shape[0],
                                        slices.shape[1],
                                        deviceId)

    assert (cpu_indices == gpu_indices_sp).all(), "CPU and GPU_SP indices don't match!"
    assert (cpu_indices == gpu_indices_dp).all(), "CPU and GPU_DP indices don't match!"

if __name__ == '__main__':
    compare_cpu_gpu_knn()

