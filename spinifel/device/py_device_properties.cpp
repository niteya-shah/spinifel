#include <cstdint>  // used by intptr_t
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>


namespace py = pybind11;


template <class T>
class ptr_wrapper {
    public:
        ptr_wrapper()
        : ptr(nullptr), safe(false)
        {}

        ptr_wrapper(T * ptr, bool is_safe=false)
        : ptr(ptr), safe(is_safe)
        {}

        ptr_wrapper(const ptr_wrapper & other)
        : ptr(other.ptr), safe(other.is_safe())
        {}

        // Allocator
        void create(size_t N) {
            ptr = new T[N];
            safe = true;
        }

        // Pointer-like accessor functions
        T & operator* () const { return * ptr; }
        T * operator->() const { return   ptr; }
        T & operator[](std::size_t idx) const { return ptr[idx]; }

        // Accessor function
        T * get() const {
            return ptr;
        }

        // Conversion function 
        operator intptr_t() const {
            // Use intptr_t to ensure that the destination type (intptr_t) is
            // big enough to hold the pointer.
            return (intptr_t) ptr;
        }

        // Deallocator
        void destroy() {
            delete ptr;
            safe = false;
        }

        // Return safety status of pointer
        bool is_safe() const { return safe; }

        // For debugging: print the pointer address
        void print_address() {
            printf("Address of ptr is %p\n", (void *) ptr);
        }

    private:
        T * ptr;
        bool safe;
};


template <class T>
struct obj_wrapper {
    T _obj;

    obj_wrapper(T & a_obj) : _obj(a_obj) {}
    obj_wrapper(T   a_obj) : _obj(a_obj) {}
    T & operator* () const { return _obj; }
    T & operator* ()       { return _obj; }
};


struct CudaError : public obj_wrapper<cudaError_t> {

    CudaError(int a_error) : obj_wrapper(static_cast<cudaError_t>(a_error)) {};

    int as_int() const;
};


int CudaError::as_int() const {
    return static_cast<int>(_obj);
}


std::string mem_to_string(const void * address, size_t size) {
    const unsigned char * p = (const unsigned char *) address;
    char buffer[size];
    for (size_t i = 0; i < size; i++) {
        snprintf(buffer + i, sizeof(buffer), "%02hhx", p[i]);
    }

    // std::string s = "";
    // for (size_t i = 0; i < size; i++) {
    //     s = s + buffer[i];
    // }
    std::string s(buffer);
    return s;
}


class DeviceProperties {
    public:
        DeviceProperties(int i) {
            cudaGetDeviceProperties(& prop, i);
        }
        ~DeviceProperties() {};

        cudaDeviceProp & operator* () { return prop; }
        cudaDeviceProp * get() { return & prop; }
        cudaError_t last_status() const { return status; }
    private:
        cudaDeviceProp prop;
        cudaError_t status;
};


PYBIND11_MODULE(py_device_properties, m) {

    py::class_<CudaError>(m, "cudaError_t")
        .def(py::init<int>())
        .def("as_int", & CudaError::as_int)
        .def("__repr__",
            [](const CudaError & a) {
                return "<CudaError: 'code=" + std::to_string(a.as_int()) + "'>";
            }
        );

    // This needs to be defined so that the ptr_wrapper has something to return
    py::class_<ptr_wrapper<cudaDeviceProp>>(m, "_CudaDeviceProp__ptr");

    py::class_<DeviceProperties>(m, "cudaDeviceProp")
        .def(py::init<int>())
        .def("get",
            [](DeviceProperties & a) {
                return ptr_wrapper<cudaDeviceProp>(a.get());
            }
        )
        .def("name",
            [](DeviceProperties & a) {
                std::string s(a.get()->name);
                return s;
            }
        )
        .def("uuid",
            [](DeviceProperties & a) {
                std::string s = mem_to_string(
                    reinterpret_cast<void *>(& a.get()->uuid), 16
                );
                return s;
            }
        )
        .def("pciBusID",
            [](DeviceProperties & a) {
                return a.get()->pciBusID;
            }
        )
        .def("pciDeviceID",
            [](DeviceProperties & a) {
                return a.get()->pciBusID;
            }
        )
        .def("pciDomainID",
            [](DeviceProperties & a) {
                return a.get()->pciBusID;
            }
        )
        .def("last_status",
            [](const DeviceProperties & a) {
                return CudaError(a.last_status());
            }
        );

        m.def(
            "cudaGetDeviceCount",
            []() {
                int device;
                cudaError_t err = cudaGetDeviceCount(& device);
                return std::make_tuple(device, CudaError(err));
            }
        );
}
