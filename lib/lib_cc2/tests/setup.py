#!/usr/bin/python
# -*- coding: utf-8 *-*
import matplotlib as mpl
mpl.use('Agg')
import pylab as plt
import numpy as np
import lib_cc2


frequencies = np.logspace(-3, 3, 40)
cc = lib_cc2.cc_res(frequencies)

# one or ore spectra
pars1 = np.array([100, 0.1, 0.04, 0.7])
pars2 = np.array([100,
                  0.1, 0.15,
                  0.04, 0.0004,
                  0.7, 1.0])

# test vectorisation
pars3 = np.array([[100, 0.1, 0.04, 0.7],
                  [100, 0.1, 0.04, 0.7],
                  [10, 0.05, 0.04, 0.3]])

pars4 = np.array([[100,
                   0.1, 0.15,
                   0.04, 0.0004,
                   0.7, 1.0],
                  [100,
                   0.1, 0.15,
                   0.04, 0.0004,
                   0.7, 1.0],
                  [100,
                   0.1, 0.15,
                   0.04, 0.0004,
                   0.7, 1.0]])

for p in (pars1, pars2, pars3, pars4):
    print 'Parametershape:', p.shape
