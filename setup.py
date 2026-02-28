import os
import sys
import subprocess
import shutil
import glob
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import torch


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # Detect CUDA
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get(
            "CUDA_HOME_PATH") or "/usr/local/cuda"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_home}",
            "-DBUILD_PYBIND=ON",
            "-DBUILD_TESTS=OFF",
            "-DFETCH_LIBTORCH=OFF",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]

        # Release build by default for PyPI
        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")

        build_args = ["--config", build_type]

        # Parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # Use fewer cores for build to avoid OOM
            build_args += ["-j4"]

        print(f"Building extension {ext.name}")
        print(f"CMake args: {' '.join(cmake_args)}")
        print(f"Build args: {' '.join(build_args)}")

        # Configure
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )

        # Build
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )

        # Find the built .so file and copy it to the correct location
        built_files = glob.glob(str(build_temp / "gdn_cuda*.so"))
        if not built_files:
            raise RuntimeError(
                f"Built .so file not found in {build_temp}. "
                f"Available files: {list(build_temp.glob('*'))}"
            )

        print(f"Found built extension: {built_files[0]}")
        extdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_files[0], ext_fullpath)
        print(f"Copied to: {ext_fullpath}")


setup(
    ext_modules=[CMakeExtension("gdn_cuda")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
