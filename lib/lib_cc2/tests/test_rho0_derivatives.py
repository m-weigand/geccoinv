#!/usr/bin/python
from setup import *

# for par in (pars1, pars2, pars3, pars4):
#     print 'pars', par
#     print 'dre_drho0', cc.dre_drho0(par).shape
#     print 'dre_dlog10rho0', cc.dre_dlog10rho0(par).shape
#     print 'dim_drho0', cc.dim_drho0(par).shape
#     print 'dim_dlog10rho0', cc.dim_dlog10rho0(par).shape

def log10im(pars):
    pars_lin = 10 ** pars
    response = cc.imag(pars_lin)
    return response

import numdifftools as nd

print dir(nd)

print 'pars2', pars2
J_num = nd.Jacobian(log10im, order=4)
dim_dlog10rho0_num = J_num(np.log10(pars2))[:, 0]
dim_dlog10rho0 = cc.dim_dlog10rho0(pars2)


print dim_dlog10rho0_num - dim_dlog10rho0
