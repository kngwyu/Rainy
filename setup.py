import io
import re
from setuptools import setup, find_packages
import sys


py_major = sys.version_info.major
py_minor = sys.version_info.minor
if py_major != 3 or py_minor <= 5:
    print('This package is only compatible with Python>=3.6, but you are running '
          'Python {}.{}. The installation will likely fail.'.format(py_major, py_minor))

with io.open('rainy/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r"__version__ = \'(.*?)\'", f.read()).group(1)


description = '''
Rainy
=====
Reinforcement learning utilities and algrithm implementations using PyTorch.

Please see https://github.com/kngwyu/Rainy for detail.
'''


requirements = [
    'click>=7.0',
    'GitPython>=2.0',
    'gym[atari]>=0.11.0',
    'numpy>=1.15.0',
    'opencv-python>=3.4',
    'Pillow>=5.0',
    'torch>=1.0',
]
test_requirements = ['pytest>=3.0']
extra_requirements = {
    'ipython': ['ipython>=7.0', 'matplotlib>=3.0', 'ptpython>=2.0'],
    'bullet': ['pybullet>=2.4']
}

setup(
    name='rainy',
    version=version,
    url='https://github.com/kngwyu/Rainy',
    project_urls={
        'Code': 'https://github.com/kngwyu/Rainy',
        'Issue tracker': 'https://github.com/kngwyu/Rainy/issues',
    },
    author='Yuji Kanagawa',
    author_email='yuji.kngw.80s.revive@gmail.com',
    description='Algorithm and utilities for deep reinforcement learning',
    long_description=description,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=requirements,
    test_requires=test_requirements,
    extras_require=extra_requirements,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
