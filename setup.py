#!/usr/bin/env python
from setuptools import setup
import sys
# from setuptools import find_packages
# find_packages

# under windows, run
# python.exe setup.py bdist --format msi
# to create a windows installer

# TODO: understand the inclusion of requirements
version_short = '0.8'
version_long = '0.8.5'

extra = {}
if sys.version_info >= (3,):
    print('V3')
    extra['use_2to3'] = True

if __name__ == '__main__':
    setup(name='geccoinv',
          version=version_long,
          description=''.join((
              'Multi dimensional geophysical inversion framework. ',
              'Note: only for use with the ccd package. If you are looking ',
              'for a general inversion framework, try pyGimli ',
              '(www.pygimli.org)',
          )),
          author='Maximilian Weigand',
          author_email='mweigand@geo.uni-bonn.de',
          url='http://www.geo.uni-bonn.de',
          # find_packages() somehow does not work under Win7 when creating a
          # msi # installer
          # packages=find_packages(),
          package_dir={'': 'lib'},
          packages=[
              'NDimInv',
              'sip_formats',
          ],
          scripts=[],
          install_requires=['numpy', 'scipy>=0.12', 'matplotlib'],
          **extra
          )
