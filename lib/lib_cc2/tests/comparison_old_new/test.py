#!/usr/bin/python
import numpy as np
import lib_cc2
import lib_dd.main

data = np.loadtxt('data.dat')
frequencies = np.loadtxt('frequencies.dat')

settings = {'global_prefix': 'spec_000_',
            'tausel': 'data_ext',
            'frequencies': frequencies,
        'Nd': 20,
        'norm_factors': 2.5650970211688082,
        'max_iterations': 20}


# old object
old = lib_dd.main.get('log10rho0log10m', settings)

# new object
settings['c'] = 1.0
new = lib_cc2.decomposition_resistivity(settings)

# forward modelling
rho0 = 10
m = np.ones(old.tau.size) * 1e-4

pars_lin = np.hstack((rho0, m))
pars_log = np.log10(pars_lin)
print pars_log.shape

f_old = old.forward(pars_log)
f_new = old.forward(pars_log)

print f_old.shape
print f_new.shape

diff = f_old - f_new
print diff

J_old = old.Jacobian(pars_log)
J_new = new.Jacobian(pars_log)
print J_old.shape, J_new.shape
diff_J = J_old - J_new
print 'Jdiff', diff_J

print J_old[0, 0]
print J_new[0, 0]


tau_old = old.tau
tau_new = new.tau
print 'tau_diff', tau_old - tau_new

# m_old = old.estimate_starting_parameters(spectrum)


# numdifftools on old Jacobian
import numdifftools as nd

print 'old'
Jac_func = lambda x: old.Jacobian(x)
Jnum_old = nd.Jacobian(Jac_func)
J_old_num = Jnum_old(pars_log)
print 'num shape', J_old_num.shape
diff = J_old - J_old_num
print 'Numdiff', diff

print('new')
Jac_func = lambda x: new.Jacobian(x)
Jnum_new = nd.Jacobian(Jac_func)
J_new_num = Jnum_new(pars_log)
print 'num shape', J_new_num.shape
diff = J_new - J_new_num
print 'Numdiff', diff
