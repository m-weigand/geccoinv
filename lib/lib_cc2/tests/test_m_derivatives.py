#!/usr/bin/python
from setup import *

for par in (pars1, pars2, pars3, pars4):
    print 'pars', par
    print 'dre_dm', cc.dre_dm(par).shape
    print 'dim_dm', cc.dim_dm(par).shape
