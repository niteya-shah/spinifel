from spack.package import *


class PySpinifel(BundlePackage):
    """This package contains **dependencies** for Spinifel. There is no actual code here."""

    version("0.1")

    depends_on("python@3.8")

    # Direct dependencies
    depends_on("py-matplotlib")
    depends_on("py-numpy@1.22")
    depends_on("py-scipy")
    depends_on("py-cupy@12.1")
    depends_on("py-mpi4py")
    depends_on("py-pytest")
    depends_on("py-toml")
    depends_on("py-h5py")
    depends_on("py-mrcfile")
    depends_on("py-pynvtx")
    depends_on("py-callmonitor")
    depends_on("py-tqdm")
    depends_on("legion@79ef214c877b9856f0961ea68b446c0d08ef35ef=cr ~cuda ~rocm +python +bindings +shared +openmp +hdf5 max_dims=4 max_num_nodes=4096 network=gasnet")
    depends_on("fftw ~mpi +openmp")

    # Transitive dependencies

    # FINUFFT / PybindGPU
    depends_on("py-pybind11@2.10")

    # skopi
    depends_on("py-scikit-learn")

    # cufinufft
    depends_on("py-six")

    # lcls2
    depends_on("cmake")
    depends_on("py-cython")
    # FIXME: not sure what Spack's name for Conda's mongodb is
    depends_on("mongo-c-driver")
    depends_on("py-pymongo")
    depends_on("curl")
    depends_on("rapidjson")
    depends_on("py-ipython")
    depends_on("py-requests")
    depends_on("py-mypy")
    depends_on("py-prometheus-client")
    depends_on("py-amityping")
    depends_on("py-bitstruct")
    depends_on("py-lcls-krtc")

    # For now, still install pip so that we can bootstrap remaining dependencies.
    depends_on("py-pip@23")

    variant("cuda", default=False, description="Enable CUDA support.")

    # When building for CUDA, also install PyCUDA (for use with cufinufft).
    with when("+cuda"):
        depends_on("py-pycuda")
