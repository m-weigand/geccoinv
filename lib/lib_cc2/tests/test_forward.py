#!/usr/bin/python
from setup import *

def forward(pars):
    Z = cc.complex(pars)
    Zre = cc.real(pars)
    Zim = cc.imag(pars)
    Zre_ana = cc.real_analytical(pars)
    Zim_ana = cc.imag_analytical(pars)

    diff_re1 = np.real(Z) - Zre_ana
    assert(np.any(np.abs(diff_re1) < 1e-14))

    diff_im1 = np.imag(Z) - Zim_ana
    assert(np.any(np.abs(diff_im1) < 1e-14))

    # print 'Z', Z
    diff_re = np.real(Z) - Zre
    diff_im = np.imag(Z) - Zim
    # print 'diff_re', diff_re
    # print 'diff_im', diff_im

    assert(np.any(np.abs(diff_re) < 1e-14))
    assert(np.any(np.abs(diff_im) < 1e-14))
    print('Check')

forward(pars1)
forward(pars2)
forward(pars3)
forward(pars4)
