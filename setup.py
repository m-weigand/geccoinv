#!/usr/bin/env python
from setuptools import setup
import sys
import os
import subprocess
# from setuptools import find_packages
# find_packages

# under windows, run
# python.exe setup.py bdist --format msi
# to create a windows installer

# TODO: understand the inclusion of requirements

version_short = '0.5'
version_long = '0.5.4'
# if we are in a git directory, use the last git commit as the version
cmd = 'git log -1 --format=%H'
try:
    if os.path.isdir('.git'):
        git_output = subprocess.check_output(cmd, shell=True).strip()
        version_long += '+' + git_output
except:
    pass

extra = {}
if sys.version_info >= (3,):
    print('V3')
    extra['use_2to3'] = True

if __name__ == '__main__':
    setup(name='geccoinv',
          version=version_long,
          description='Multi dimensional geophysical inversion framework',
          author='Maximilian Weigand',
          author_email='mweigand@geo.uni-bonn.de',
          url='http://www.geo.uni-bonn.de',
          # find_packages() somehow does not work under Win7 when creating a
          # msi # installer
          # packages=find_packages(),
          package_dir={'': 'lib'},
          packages=['lib_cc',
                    'lib_cc2',
                    'lib_cc_conductivity',
                    'NDimInv',
                    'sip_formats'],
          scripts=[],
          install_requires=['numpy', 'scipy>=0.12', 'matplotlib'],
          **extra
          )
