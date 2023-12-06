from setuptools import setup
from torch.utils import cpp_extension
 
setup(
    name='distributed_gloo_cpp',						# 编译后的链接库名称
    ext_modules=[
        cpp_extension.CppExtension(
            'distributed_gloo_cpp', ['galaxy/core/csrc/DistributedGloo.cpp'],  # name, source 
            # extra_compile_args=['-DUSE_C10D_GLOO']  # 添加宏定义，启用GLOO模块
        ),
    ],
    cmdclass={						       # 执行编译命令设置
        'build_ext': cpp_extension.BuildExtension
    }
)