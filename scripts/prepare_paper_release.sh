#!/bin/bash
# create a tar.gz file containing all files for distribution with the paper
test e GeccoInv.tar && rm GeccoInv.tar
test e GeccoInv.tar.gz && rm GeccoInv.tar.gz
tar cvf GeccoInv.tar --exclude 'docs/Presentations/*' --exclude 'docs/Literature/*'  setup.py src/ lib/ Examples/ docs/ LICENCE  Readme.rst
gzip GeccoInv.tar
