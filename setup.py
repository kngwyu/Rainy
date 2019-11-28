import io
import os
import re
from setuptools import find_packages, setup


NAME = "rainy"
AUTHOR = "Yuji Kanagawa"
EMAIL = "yuji.kngw.80s.revive@gmail.com"
URL = "https://github.com/kngwyu/Rainy"
REQUIRES_PYTHON = ">=3.6.0"
DESCRIPTION = "Algorithm and utilities for deep reinforcement learning"

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "rainy/__init__.py"), "rt", encoding="utf8") as f:
    VERSION = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION


REQUIRED = [
    "click>=7.0",
    "colorama>=0.4",
    "GitPython>=2.0",
    "gym[atari]>=0.11.0",
    "numpy>=1.15.0",
    "opencv-python>=3.4",
    "pandas>=0.25",
    "Pillow>=6.2",
    "torch>=1.0",
]
EXTRA = {
    "bullet": ["pybullet>=2.4"],
    "horovod": ["horovod>=0.16"],
}


setup(
    name=NAME,
    version=VERSION,
    url=URL,
    project_urls={"Code": URL, "Issue tracker": URL + "/issues",},
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require=EXTRA,
    license="Apache2",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)
