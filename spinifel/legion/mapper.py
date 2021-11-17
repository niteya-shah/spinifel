from __future__ import print_function

import cffi
import pygion
import os
import subprocess

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
simple_mapper_h_path = os.path.join(root_dir, 'legion_mapper', 'simple_mapper.h')
simple_mapper_header = subprocess.check_output(['gcc', '-E', '-P', simple_mapper_h_path]).decode('utf-8')
legion_mappers_so_path = os.path.join(root_dir, 'legion_mapper', 'build', 'liblegion_mappers.so')

ffi = pygion.ffi
ffi.cdef(simple_mapper_header)
c = ffi.dlopen(legion_mappers_so_path)

c.register_simple_mapper()
