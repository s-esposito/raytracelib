import os
import pybind11
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise RuntimeError("PyTorch must be installed before running setup.py.")

import shutil

if shutil.which("nvcc") is None:
    raise RuntimeError("CUDA compiler (nvcc) is not found! Ensure CUDA is installed.")

_src_path = os.path.dirname(os.path.abspath(__file__))


def get_eigen_include():
    EIGEN_WEB_URL = (
        "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2"
    )
    TMP_EIGEN_FILE = "tmp_eigen.tar.bz2"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    # eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

    # if eigen_include_dir is not None:
    #     print("EIGEN3_INCLUDE_DIR is set: ", eigen_include_dir)
    #     return eigen_include_dir

    target_dir = os.path.join(_src_path, "ext", EIGEN3_DIRNAME)
    if os.path.exists(target_dir):
        return target_dir
    else:
        import urllib.request

        print("Couldn't find Eigen locally, downloading...")
        req = urllib.request.Request(
            EIGEN_WEB_URL,
            data=None,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36"
            },
        )

        # make ext dir
        if not os.path.exists(os.path.join(_src_path, "ext")):
            os.mkdir(os.path.join(_src_path, "ext"))

        with urllib.request.urlopen(req) as resp, open(
            os.path.join(_src_path, "ext", TMP_EIGEN_FILE), "wb"
        ) as file:
            data = resp.read()
            file.write(data)
        import tarfile

        tar = tarfile.open(os.path.join(_src_path, "ext", TMP_EIGEN_FILE))
        tar.extractall(path=os.path.join(_src_path, "ext"))
        tar.close()

        os.remove(os.path.join(_src_path, "ext", TMP_EIGEN_FILE))

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
                    # "sphere.cu",
                    "bvh.cu",
                    "raytracer.cu",
                    "bindings.cpp",
                ]
            ],
            include_dirs=[
                os.path.join(_src_path, "include"),
                get_eigen_include(),
                pybind11.get_include(),
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
        "pybind11[global]",
    ],
    install_requires=[
        "torch>=2.1.0",
    ],
)
