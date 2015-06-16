#!/usr/bin/python
from setup import *

def forward(pars):
    Z = cc.complex(pars)
    Zre = cc.real(pars)
    Zim = cc.imag(pars)

    # print 'Z', Z
    diff_re = np.real(Z) - Zre
    diff_im = np.imag(Z) - Zim
    # print 'diff_re', diff_re
    # print 'diff_im', diff_im

    assert(np.any(np.abs(diff_re) < 1e-14))
    assert(np.any(np.abs(diff_im) < 1e-14))

forward(pars1)
forward(pars2)
forward(pars3)
forward(pars4)
