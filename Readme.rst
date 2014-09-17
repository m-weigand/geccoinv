Debye Decomposition routines written in Python
==============================================

Licence
-------

This program is licenced under the GPL3 or later licence. See LICENCE.txt and
headers of individual files for more information.

Requirements
------------

Installation
------------

Please refer to the documentation found in docs/doc

Quick notes:

::

    python setup.py build
    python setup.py install --prefix=$HOME/inst/dd

    export PYTHONUSERBASE=$HOME/inst/pip_installs
    export PYTHONPATH=$HOME/inst/pip_installs/lib/python2.7/\
        site-packages/:$PYTHONPATH
    python setup.py install --user
    export PATH=$HOME/inst/pip_installs/bin:$PATH
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

