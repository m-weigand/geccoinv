#!/usr/bin/python
from setup import *

for par in (pars1, pars2, pars3, pars4):
    print 'pars', par
    print 'dre_dtau', cc.dre_dtau(par).shape
    print 'dre_dlog10tau', cc.dre_dlog10tau(par).shape
    print 'dim_dtau', cc.dim_dtau(par).shape
    print 'dim_dlog10tau', cc.dim_dlog10tau(par).shape
