GeccoInv - N-dimensional geophysical inversion framework
========================================================

Licence
-------

This program is licenced under the GPL3 or later licence. See LICENCE.txt and
headers of individual files for more information.

Requirements
------------

Latex is required for plot output.

The following Python packages are required to run the Debye decomposition
routines:

numpy
scipy
matplotlib

In order to build the documentation, the additional packages are required:

sphinx

Under Debian-related linux systems, these packages can be installed using the
commands:

    sudo apt-get install texlive-latex-base texlive-latex-extra\
        texlive-fonts-recommended texlive-fonts-extra dvipng
    sudo apt-get install python2.7-numpy python-matplotlib python2.7-scipy\
        python-setuptools python-nose python-tornado

In order to build the documentation, the following sphinx-related packages must
be installed:

    sudo apt-get install python-sphinx
    sudo apt-get install python-sphinx python-sphinxcontrib.blockdiag
    pip install sphinxcontrib-programoutput
    pip install numpydoc

Under Windows, the program was tested using the 'pythonxy' distribution
(https://code.google.com/p/pythonxy/). Additionally, latex support ist required
for all plots. The programs was tested with the Miktex distribution
(http://www.miktex.org/).


Installation
------------

::

    python setup.py build
    python setup.py install


The package can also be installed to a user-defined directory:

::
    export PYTHONUSERBASE=$HOME/inst/pip_installs
    export PYTHONPATH=$HOME/inst/pip_installs/lib/python2.7/\
        site-packages/:$PYTHONPATH
    python setup.py install --user
    export PATH=$HOME/inst/pip_installs/bin:$PATH


For developers:

::
    python seutp.py develop --user

To build the documentation

::

    cd docs/doc
    python setup.py sphinx_build

Setuptools Developer Guide:

https://pythonhosted.org/setuptools/setuptools.html

Documentation
-------------
 * Documentation can be found in docs/
 * The sphinx-generated documentation can be found in docs/doc
 * For the internal version related literature can be found in docs/literature

