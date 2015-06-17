#!/usr/bin/python
import numpy as np

import lib_cc2.decomposition_res as cc_dec_res

# load data
ddir = 'results/stats_and_rms/'
frequencies = np.loadtxt(ddir + '../frequencies.dat')
tau = np.loadtxt(ddir + '../tau.dat')
m = np.loadtxt(ddir + 'm_i_results.dat')
rho0 = np.atleast_1d(np.loadtxt(ddir + 'rho0_results.dat'))


print frequencies.shape
print tau.shape
print rho0.shape
print m.shape


# the decomposition object

settings = {
    'frequencies': frequencies,
    'tau': tau,
    'c': 1.0
}
ccd = cc_dec_res.decomposition_resistivity(settings)

# test forward modelling
pars = np.hstack((rho0, m))
erg = ccd.forward(pars)

def plot():
    from crlab_py.mpl import *
    fig, axes = plt.subplots(2, 1, figsize=(5, 4))
    ax = axes[0]
    ax.loglog(frequencies, erg[0, :], '.-')
    ax = axes[1]
    ax.loglog(frequencies, -erg[1, :], '.-')
    fig.savefig('output.png', dpi=300)


# test Jacobian
J = ccd.J(pars).squeeze()
print 'J', J.shape
import numdifftools
Jfunnum = numdifftools.Jacobian(ccd.forward)
Jnum = Jfunnum(pars)
print 'Jnum', Jnum.shape

# compare real derivatives
Jre = J[0:163, :]
Jrenum = Jnum[0:25, :].T
diff = Jre - Jrenum
print diff.min(), diff.max()


# compare imag derivatives
Jim = J[163:, :]
Jimnum = Jnum[25:, :].T
diff = Jim - Jimnum
print diff.min(), diff.max()
