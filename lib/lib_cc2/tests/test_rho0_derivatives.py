#!/usr/bin/python
from setup import *

for par in (pars1, pars2, pars3, pars4):
    print 'pars', par
    print 'dre_drho0', cc.dre_drho0(par).shape
    print 'dre_dlog10rho0', cc.dre_dlog10rho0(par).shape
    print 'dim_drho0', cc.dim_drho0(par).shape
    print 'dim_dlog10rho0', cc.dim_dlog10rho0(par).shape
