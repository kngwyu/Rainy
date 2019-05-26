import io
import os
import re
from setuptools import Command, find_packages, setup
from shutil import rmtree


NAME = 'rainy'
AUTHOR = 'Yuji Kanagawa'
EMAIL = 'yuji.kngw.80s.revive@gmail.com'
URL = 'https://github.com/kngwyu/Rainy'
REQUIRES_PYTHON = '>=3.6.0'
DESCRIPTION = 'Algorithm and utilities for deep reinforcement learning'

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'rainy/__init__.py'), 'rt', encoding='utf8') as f:
    VERSION = re.search(r"__version__ = \'(.*?)\'", f.read()).group(1)

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION


REQUIRED = [
    'click>=7.0',
    'GitPython>=2.0',
    'gym[atari]>=0.11.0',
    'numpy>=1.15.0',
    'opencv-python>=3.4',
    'Pillow>=5.0',
    'torch>=1.0',
]
TEST = ['pytest>=3.0']
EXTRA = {
    'ipython': ['ipython>=7.0', 'matplotlib>=3.0', 'ptpython>=2.0'],
    'bullet': ['pybullet>=2.4']
}


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')
        sys.exit()


setup(
    name=NAME,
    version=VERSION,
    url=URL,
    project_urls={
        'Code': URL,
        'Issue tracker': URL + '/issues',
    },
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    test_requires=TEST,
    extras_require=EXTRA,
    license='Apache2',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
