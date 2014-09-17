#!/bin/bash
if [ ! $# -eq 2 ]; then
    echo "Need two arguments: Short version and long version"
    exit
fi


version_short=$1
version_long=$2

# set version in setup.py
sed -i "s/version_short = .*$/version_short = '${version_short}'/" setup.py
sed -i "s/version_long = .*$/version_long = '${version_long}'/" setup.py

echo "version = '${version_long}'" > lib/NDimInv/version.py
