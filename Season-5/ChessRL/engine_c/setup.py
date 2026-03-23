"""Build script for the C++ Chinese Chess engine."""

import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

compile_args = ["-O3", "-DNDEBUG", "-std=c++17"]
if platform.machine() == "arm64":
    compile_args.append("-mcpu=apple-m1")
else:
    compile_args.append("-march=native")

ext_modules = [
    Pybind11Extension(
        "engine_c._xiangqi",
        ["xiangqi.cpp", "bindings.cpp"],
        extra_compile_args=compile_args,
        include_dirs=["."],
    ),
]

setup(
    name="engine_c",
    version="0.1.0",
    packages=["engine_c"],
    package_dir={"engine_c": "."},
    package_data={"engine_c": ["*.pyi"]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
