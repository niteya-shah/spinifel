CC   ?= gcc
CXX  ?= g++
NVCC ?= nvcc


CFLAGS    ?= -fPIC -O3 -funroll-loops -march=native
CXXFLAGS  ?= $(CFLAGS) -std=c++14
NVCCFLAGS ?= -std=c++14 -ccbin=$(CXX) -O3 -Xcompiler "$(CXXFLAGS)"

PYVER := $(shell                                                           \
    python -c "import sys; print('{}.{}'.format(*sys.version_info[0:2]))"  \
)
PYINCLUDES := $(shell python -m pybind11 --includes)
PYSUFFIX   := $(shell python$(PYVER)-config --extension-suffix)

CUDA_INC += $(INC) -I../util

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.c
	$(CC) -c $(CFLAGS) $(INC) $< -o $@
%.o: %.cu
	$(NVCC) --device-c -c $(NVCCFLAGS) $(CUDA_INC) $< -o $@

%:%.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

py_device_properties$(PYSUFFIX): py_device_properties.cpp
	$(NVCC) -shared $(NVCCFLAGS) $(PYINCLUDES) $^ -o $@

default: all

all: py_device_properties$(PYSUFFIX)

clean:
	rm py_device_properties$(PYSUFFIX)