.. GeccoInv documentation master file, created by
   sphinx-quickstart on Mon Jun 17 09:29:21 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
    :hidden:

    tutorial1
    design_overview
    inversion
    debugging

========
GeccoInv
========

Introduction
============

GeccoInv is a pure python implentation of an n-dimensional Tikhonov-inversion
scheme. It was primarily built for the implementation of the Debye
decomposition scheme to describe electrical spectral induced polarisation (SIP)
signatures but also be used for arbitrary inversion problems. It supports
multiple regularisation schemes (smoothing operators of first and second order)
and various methods of determining regularisation (:math:`\lambda`) parameters.

Feedback, bug reports, and general advice is always welcome. Please use the
issue tracker on the github page for communication
(https://github.com/m-weigand/geccoinv).  Before you report a bug make sure to
look through the existing bugs; perhaps your problem was already reported. Also
make sure to include all necessary information to reproduce the bug. A small
example data set and the command line options to trigger the bug are always
helpfull!

Design Overview
===============

:doc:`design_overview`

Build you own inversion
=======================

:doc:`tutorial1`

Inversion
=========

:doc:`inversion`

Debugging
==========

:doc:`debugging`

Copyright
=========

The GeccoInv project is licenced under the GPL 3 or later
(https://www.gnu.org/licenses/licenses.html#GPL).

TODO: Documentation licence.

Installation
============

The *setuptools* distribution tools to manage the installation procedure:

::

    python setup.py install

should suffice to install the libraries and scripts.

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

For certain versions of numpy (Debian Wheezy), there exist problems with
libopenblas (CPU goes to 100% and the program freezes). These problems are
related to multithreading issues in Python

Workarounds:

* Use only one thread in openblas:
  ::

    OPENBLAS_NUM_THREADS=1 dd_single.py [...]

* switch to atlas/libblas:

  ::

    update-alternatives --config libblas.so
    update-alternatives --config libblas.so.3

Windows
-------

 * For a complete Python distribution, use Python(x,y) 2.7.x
   (https://code.google.com/p/pythonxy/)


.. _ref_build_your_own_inversion:

Long-term Todo
==============

The following items are somewhat long-term todo items which could also be seen
as a kind of general development directives.

Organisational things
---------------------

* Explore "interactive" fitting using IPython
* Clean up interface
* Clean up test cases

Functionality
-------------

* We need to check for numdifftools in order to run the unit tests!
* Reach full test coverage
* logfile should be in order (even for multiprocessing, perhaps merge logfiles later?)

Long-term:

* Implement c not equal 1 routines
* GSVD solution for Tikhonov first order reg.
* Determine lambda from the l-curve
* Implement univariate lambda search
* SVD - analysis

Profiling and Debugging
-----------------------

At least in *dd_singly.py* we import the `memory_profiler` module. Just add the
decorator `@profile` in front of a function and during execution a line-by-line
memory usage profile will be printed to STDOUT.

https://github.com/fabianp/memory_profiler

To profile the memory increase between subsequent spectra in **dd_single.py**,
add the decorator to the `fit_one_spectrum` function:

::

    @profile
    def fit_one_spectrum(opts):

Then run the dd_single command as usual, but pipe to a logfile:

::

    dd_single.py [OPTIONS] | tee logfile

Then grep for specific calls:

::

    grep "run_inversion" logfile
    206     51.9 MiB      3.5 MiB       ND.run_inversion()
    206     54.9 MiB      2.9 MiB       ND.run_inversion()
    206     55.0 MiB      0.1 MiB       ND.run_inversion()
    206     55.1 MiB      0.1 MiB       ND.run_inversion()
    206     55.1 MiB      0.1 MiB       ND.run_inversion()
    206     55.2 MiB      0.1 MiB       ND.run_inversion()
    206     55.2 MiB      0.1 MiB       ND.run_inversion()
    206     55.3 MiB      0.0 MiB       ND.run_inversion()
    206     55.4 MiB      0.1 MiB       ND.run_inversion()
    206     55.4 MiB      0.1 MiB       ND.run_inversion()

As a side note, it would be nice to get `pycallgraph` to run:

http://pycallgraph.slowchop.com/en/master/guide/command_line_usage.html

We could also use Travis (https://travis-ci.org/) von continous integration.

Notes
-----

* General goal: Resistivity and conductivity formulation, transformation from
  continuous to discrete formulation, inclusion of c (Cole-Cole case)

* The explicit computation of the matrix product
  :math:`\underline{\underline{J}}^T \cdot \underline{\underline{J}}` is
  computationally expensive. When the CG solver is used most of the time this
  product can be prevented by performing the matrix-vector product from right
  to left.

API
===

.. toctree::
    :maxdepth: 0

    api/NDimInv
    api/lib_cc
    api/lib_cc
    api/sip_formats

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
