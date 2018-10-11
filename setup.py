from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

requirements = [
    'click>=7.0',
    'GitPython>=2.0',
    'gym[atari]>=0.10.5',
    'numpy>=1.0',
    'opencv-python>=3.4',
    'torch>=0.4',
    'torchvision>=0.2.1',
]
test_requirements = ['pytest>=3.0']
print(requirements)
setup(name='rainy',
      author='Yuji Kanagawa',
      url='https://github.com/kngwyu/Rainy',
      version='0.1',
      packages=find_packages(),
      install_requires=requirements,
      test_requires=test_requirements)

