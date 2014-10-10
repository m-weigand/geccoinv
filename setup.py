#!/usr/bin/env python
from setuptools import setup
# from setuptools import find_packages
# find_packages

# under windows, run
# python.exe setup.py bdist --format msi
# to create a windows installer

# TODO: understand the inclusion of requirements

version_short = '0.5'
version_long = '0.5.4'

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
                    'NDimInv',
                    'sip_formats'],
          scripts=[],
          install_requires=['numpy', 'scipy>=0.12', 'matplotlib']
          )
