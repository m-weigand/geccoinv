Debugging new inversion routines
--------------------------------

Sometimes inversions fail for some reasons and this page collects various ways
to check the different aspects of the inversion:

* Check the starting model! It is of major importance to find good heuristics
  to create suitable starting models.
* Check the residuals for each iteration
* Check the data normalisation: Especially data weighting factors, the
  regularisation, and the absolute data normalization are directly influenced
  by each other.

Iteration._update_dict
----------------------

Each iteration can store information about the inversion process leading to the
next iteration. However, this requires a lot of memory (RAM), and is thus by
default turned off. To active it, use the environment variable
`NDIMINV_RETAIN_DETAILS=1`:

::

    NDIMINV_RETAIN_DETAILS=1 inversion_routine.py

Except for the final iteration (which didn't compute any data for the next
iteration), each Iteration now has the dict `._update_dict` which holds all
data necessary to reproduce the inversion step.

::

    ND = NDimInv(...)
    ND.run_inversion()
    print ND.iterations[0]._update_dict.keys()


