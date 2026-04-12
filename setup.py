from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mnistocr", 
        ["src/logistic.cu", "src/bindings.cu"],
        include_dirs=["src"],
        language="c++"
    ),
]

setup(
    name="mnistocr",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},

    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},

    install_requires=[
        "numpy"
    ],

    entry_points={
        "console_scripts": [
            "mnistocr=mnistocr.cli:main"
        ]
    },
)