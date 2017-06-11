import os
import torch
from torch.utils.ffi import create_extension

#this_file = os.path.dirname(__file__)

sources = ['c_src/warp_lib.c']
headers = ['c_src/warp_lib.h']
defines = []
with_cuda = False
extra_objects = None

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/my_lib_cuda.c']
    headers += ['src/my_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
    this_file = os.path.dirname(os.path.realpath(__file__))
    extra_objects = ['src/my_lib_cuda_kernel.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.warp_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
