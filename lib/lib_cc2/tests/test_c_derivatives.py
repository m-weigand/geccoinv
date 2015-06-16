#!/usr/bin/python
from setup import *

for par in (pars1, pars2, pars3, pars4):
    print 'pars', par
    print 'dre_dc', cc.dre_dc(par).shape
    print 'dim_dc', cc.dim_dc(par).shape
