from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

install_requires = [
    'baselines>=0.1.5',
    'GitPython>=2.0',
    'gym[atari]>=0.10.5',
    'numpy>=1.0',
    'torch>=0.4.0',
    'torchvision>=0.2.1',
]

test_requires = [
    'pytest',
]

setup(name='rainy',
      author='Yuji Kanagawa',
      url='https://github.com/kngwyu/Rainy',
      version='0.1',
      packages=find_packages(),
      dependency_links=['git://github.com/kngwyu/baselines.git#egg=baselines'],
      install_requires=install_requires,
      test_requires=test_requires)

