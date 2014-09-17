#!/bin/bash
# This is some kind of reminder script to pick up Debian packaging
# apt-get install python-stdeb
rm -r dist/
python setup.py sdist
cd dist
py2dsc "Maximilian Weigand <mweigand@geo.uni-bonn.de>" *.tar.gz
cd deb_dist
cd debyedecomposition-1.0
debuild

echo "dist/deb_dist/python-debyedecomposition_1.0-1_all.deb"
