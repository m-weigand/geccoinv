.. GeccoInv documentation master file, created by
   sphinx-quickstart on Mon Jun 17 09:29:21 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
    :hidden:

    tutorial_dd_ndiminv
    design_overview
    inversion

========
GeccoInv
========

Introduction
============

This project is a collection of Python libraries and scripts which facilitate
the analysis of spectral induced polarisation (SIP) spectra using a Debye decomposition
scheme (see :doc:`dd_general` for general notes on the Debye decomposition
scheme). The basis of all scripts is formed by the *NDimInv* library which
implements a somewhat general multi-dimensional inversion scheme which supports
an arbitrary number of regularizations between various dimensions. Building
upon these, three front-ends for the Debye Decompositions are provided:

* *dd_single.py* fits an arbitrary number of CR spectra without any
  kind of regularization between the spectra.
* *dd_time.py* fits an arbitrary number of time-lapse CR spectra with a
  regularization on the time axis (separate regularization parameters for
  :math:`\rho_0` and the chargeabilities :math:`m_i`)
* *dd_space_time.py* fits and arbitrary number of time-lapse CR spectra
  recovered from imaging results with the same time regularization as applied
  by *dd_time.py*. No spatial regularization is applied.

See :ref:`ref_fit_routines` for detailed information on each of the front-ends.

Each front-end is accompanied by a post processing tool (*ddps.py*, *ddpt.py*,
and *ddpst.py*) which allows various analysis and postprocessing steps to be
applied on inversion results. This includes filtering and plotting routines.
(TODO: Link). As soon as the underlying API is stabilized, the low level
functions will be explain here: :ref:`ref_build_your_own_inversion`

We are always interested in your feedback and recommendations. Please use the
bug tracker (see next paragraph) or contact us directly under
mweigand@geo.uni-bonn.de.

Bugs and feature requests are managed by the bug tracker found here:
http://testing.geo.uni-bonn.de/trac . Before you report a bug make sure to look
through the existing bugs; perhaps your problem was already reported. Also make
sure to include all necessary information to reproduce the bug. A small example
data set and the command line options to trigger the bug are always helpfull!
Please note that the bug tracker is password protected to prevent vandalism.
Please contact mweigand@geo.uni-bonn.de for your personal access information.

Copyright
=========

The GeccoInv project is licenced under the GPL 3 or later (https://www.gnu.org/licenses/licenses.html#GPL).

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

Design Overview
===============

:doc:`design_overview`

Inversion
=========

:doc:`inversion`

.. _ref_fit_routines:

Debye Decomposition
===================

.. toctree::

    dd_general

Fit Routines
============

Three fit routines are provided with specialized input/output configurations:

.. toctree::
    :maxdepth: 1
    :hidden:

    programs/dd_single
    programs/dd_time
    programs/dd_space_time


* Fiting of separate sepctra (1D) :doc:`programs/dd_single`
* Fitting of time-regularized spectra (2D) :doc:`programs/dd_time`
* Fitting of spatially distributed time-regularized spectra (3D, but no spatial
  regularization) :doc:`programs/dd_space_time`

.. _ref_build_your_own_inversion:

Build you own inversion
=======================

:doc:`tutorial_dd_ndiminv`

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
    api/lib_dd
    api/lib_cc
    api/sip_formats

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
