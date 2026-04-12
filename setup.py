import os
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Dummy extension — CMake does the real compilation."""
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    """Runs mkdir build && cmake .. && make automatically."""

    def build_extension(self, ext):
        ext_fullpath = self.get_ext_fullpath(ext.name)
        extdir = os.path.abspath(os.path.dirname(ext_fullpath))
        source_dir = os.path.abspath(".")
        build_dir = os.path.join(self.build_temp, ext.name)

        os.makedirs(build_dir, exist_ok=True)

        subprocess.check_call(
            ["cmake", source_dir,
             f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
             "-DCMAKE_BUILD_TYPE=Release"],
            cwd=build_dir,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=build_dir,
        )


setup(
    name="mnistocr",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},

    ext_modules=[CMakeExtension("mnistocr")],
    cmdclass={"build_ext": CMakeBuild},

    install_requires=[
        "numpy"
    ],

    entry_points={
        "console_scripts": [
            "mnistocr=mnistocr.cli:main"
        ]
    },
)
