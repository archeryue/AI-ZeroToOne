"""Build script for the C++ Go engine (pybind11)."""

import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

compile_args = ["-O3", "-DNDEBUG", "-std=c++17"]
if platform.machine() in ("arm64", "aarch64"):
    compile_args.append("-mcpu=apple-m1")
else:
    compile_args.append("-march=native")

ext_modules = [
    Pybind11Extension(
        "go_engine",
        ["go.cpp", "bindings.cpp"],
        extra_compile_args=compile_args,
        include_dirs=["."],
        language="c++",
    ),
]

setup(
    name="go_engine",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
