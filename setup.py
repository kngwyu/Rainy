from pipenv.project import Project
from pipenv.utils import convert_deps_to_pip
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

pfile = Project(chdir=False).parsed_pipfile

requirements = convert_deps_to_pip(pfile['packages'], r=False)
test_requirements = convert_deps_to_pip(pfile['dev-packages'], r=False)
print(requirements)
setup(name='rainy',
      author='Yuji Kanagawa',
      url='https://github.com/kngwyu/Rainy',
      version='0.1',
      packages=find_packages(),
      install_requires=requirements,
      test_requires=test_requirements)

