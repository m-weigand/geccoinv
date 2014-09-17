#!/bin/bash
#-------------------------------------------------------------------------------
# WARNING: This is just a really ugly thing to do. If you don't have any good
# reason, just use the setup.py file to install the routines!
#-------------------------------------------------------------------------------
# Install to $HOME/bin

echo "export PYTHONPATH=/home/mweigand/Uni/Programme/DebyeDecomposition/lib:\$PYTHONPATH" >> ~/.bashrc

echo "export PYTHONPATH=/home/mweigand/Uni/Programme/DebyeDecomposition/src/dd_test:\$PYTHONPATH" >> ~/.bashrc
echo "export PYTHONPATH=/home/mweigand/Uni/Programme/DebyeDecomposition/src/dd_post:\$PYTHONPATH" >> ~/.bashrc
echo "export PYTHONPATH=/home/mweigand/Uni/Programme/DebyeDecomposition/src/debye_decomposition:\$PYTHONPATH" >> ~/.bashrc

pwdx="$PWD"
cd ~/bin
rm debye_decomposition.py dd_post.py dd_test.py
ln -s $pwdx/src/debye_decomposition/debye_decomposition.py
ln -s $pwdx/src/dd_post/dd_post.py
ln -s $pwdx/src/dd_test/dd_test.py
