"""Build script for the C++ Chinese Chess engine."""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "engine_c._xiangqi",
        ["xiangqi.cpp", "bindings.cpp"],
        extra_compile_args=["-O3", "-march=native", "-DNDEBUG", "-std=c++17"],
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
