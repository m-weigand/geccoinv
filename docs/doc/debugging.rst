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
