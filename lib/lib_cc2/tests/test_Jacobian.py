#!/usr/bin/python
from setup import *

for par in (pars1, pars2, pars3, pars4):
    print 'pars', par
    J = cc.Jacobian_re_im(par)
    print 'J.shape', J.shape

# for now we can only check the Jacobian using numdifftools if only one
# spectrum is used

print '---------------------'
# nummdiff
import numdifftools as nd

J = cc.Jacobian_re_im(pars1)
Jnum = nd.Jacobian(cc.real, order=4)
Jrenum = Jnum(pars1).T

# diff rho0
diff_re_rho0 = Jrenum[0, :] - J.squeeze()[0, :]
print 'diff_re_rho0', diff_re_rho0.min(), diff_re_rho0.max()

# diff m
diff_re_m = Jrenum[1, :] - J.squeeze()[1, :]
print 'diff_re_m', diff_re_m.min(), diff_re_m.max()

# diff tau
diff_re_tau = Jrenum[2, :] - J.squeeze()[2, :]
print 'diff_re_tau', diff_re_tau.min(), diff_re_tau.max()

# diff c
diff_re_c = Jrenum[3, :] - J.squeeze()[3, :]
print 'diff_re_c', diff_re_c.min(), diff_re_c.max()


Jnum = nd.Jacobian(cc.imag, order=4)
Jimnum = Jnum(pars1).T

# diff rho0
diff_im_rho0 = Jimnum[0, :] - J.squeeze()[4, :]
print 'diff_im_rho0', diff_im_rho0.min(), diff_im_rho0.max()

# diff m
diff_im_m = Jimnum[1, :] - J.squeeze()[5, :]
print 'diff_im_m', diff_im_m.min(), diff_im_m.max()

# diff tau
diff_im_tau = Jimnum[2, :] - J.squeeze()[6, :]
print 'diff_im_tau', diff_im_tau.min(), diff_im_tau.max()

# diff c
diff_im_c = Jimnum[3, :] - J.squeeze()[7, :]
print 'diff_im_c', diff_im_c.min(), diff_im_c.max()
