import os
import urllib.request
import tarfile
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise RuntimeError("PyTorch must be installed before running setup.py.")

import shutil

if shutil.which("nvcc") is None:
    raise RuntimeError("CUDA compiler (nvcc) is not found! Ensure CUDA is installed.")

_src_path = os.path.dirname(os.path.abspath(__file__))


def get_pybind11_include():
    PYBIND11_WEB_URL = "https://github.com/pybind/pybind11/archive/refs/tags/v2.11.0.tar.gz"
    TMP_PYBIND11_FILE = "tmp_pybind11.tar.gz"
    PYBIND11_DIRNAME = "pybind11-2.11.0"

    target_dir = os.path.join(_src_path, "ext", PYBIND11_DIRNAME)
    if os.path.exists(target_dir):
        return target_dir
    else:
        print("Couldn't find pybind11 locally, downloading...")
        req = urllib.request.Request(
            PYBIND11_WEB_URL,
            data=None,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        ext_dir = os.path.join(_src_path, "ext")
        os.makedirs(ext_dir, exist_ok=True)

        pybind11_archive_path = os.path.join(ext_dir, TMP_PYBIND11_FILE)
        with urllib.request.urlopen(req) as resp, open(pybind11_archive_path, "wb") as file:
            file.write(resp.read())

        with tarfile.open(pybind11_archive_path) as tar:
            tar.extractall(path=ext_dir)

        os.remove(pybind11_archive_path)
        return target_dir


def get_eigen_include():
    EIGEN_WEB_URL = (
        "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2"
    )
    TMP_EIGEN_FILE = "tmp_eigen.tar.bz2"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    target_dir = os.path.join(_src_path, "ext", EIGEN3_DIRNAME)
    if os.path.exists(target_dir):
        return target_dir
    else:
        print("Couldn't find Eigen locally, downloading...")
        req = urllib.request.Request(
            EIGEN_WEB_URL,
            data=None,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        ext_dir = os.path.join(_src_path, "ext")
        os.makedirs(ext_dir, exist_ok=True)

        eigen_archive_path = os.path.join(ext_dir, TMP_EIGEN_FILE)
        with urllib.request.urlopen(req) as resp, open(eigen_archive_path, "wb") as file:
            file.write(resp.read())

        with tarfile.open(eigen_archive_path) as tar:
            tar.extractall(path=ext_dir)

        os.remove(eigen_archive_path)
        return target_dir


nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

c_flags = ["-O3", "-std=c++17"]

"""
Usage:
python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)
python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)
python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)
"""
setup(
    name="raytracelib",
    version="1.0.0",
    description="CUDA RayTracer with BVH acceleration",
    url="https://github.com/avg-dev/raytrace-mesh",
    author="Stefano Esposito, Bozidar Antic",
    author_email="stefano.esposito@uni-tuebingen.de",
    ext_modules=[
        CUDAExtension(
            name="raytracing_gpu",  # extension name, import this to use CUDA API
            sources=[
                os.path.join(_src_path, "src", f)
                for f in [
                    "bvh.cu",
                    "raytracer.cu",
                    "bindings.cpp",
                ]
            ],
            include_dirs=[
                os.path.join(_src_path, "include"),
                get_eigen_include(),
                get_pybind11_include(),
            ],
            extra_compile_args={
                "cxx": c_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    setup_requires=[
        "setuptools",
    ],
    install_requires=[
        "torch>=2.1.0",
        "trimesh",
        "opencv-python",
        "numpy",
        "tqdm",
        "matplotlib",
        "dearpygui",
    ],
)