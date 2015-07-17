#!/usr/bin/python
import numpy as np
import lib_cc2
import lib_dd.main

data = np.loadtxt('data.dat')
frequencies = np.loadtxt('frequencies.dat')


settings = {'global_prefix': 'spec_000_',
            'tausel': 'data',
            'frequencies': frequencies[:],
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
m = np.ones(old.tau.size) * 1e-3

pars_lin = np.hstack((rho0, m))
pars_log = np.log10(pars_lin)
print pars_log.shape

f_old = old.forward(pars_log)
f_new = new.forward(pars_log)

print 'forward-diff', f_old - f_new
exit()
print 'reshape:', f_old.T.reshape(frequencies.size * 2)

print 'forward'
print f_old.shape
print f_new.shape

diff = f_old - f_new
print 'Min/Max forward diff', diff.min(), diff.max()

J_old = old.Jacobian(pars_log)
J_new = new.Jacobian(pars_log)
print 'shapes', J_old.shape, J_new.shape
diff_J = J_old - J_new
print 'Jdiff', diff_J

print J_old[0, 0]
print J_new[0, 0]


tau_old = old.tau
tau_new = new.tau
print 'tau_diff', tau_old - tau_new

# m_old = old.estimate_starting_parameters(spectrum)

mr_old = old.del_mim_del_dc(pars_log)
cc_pars = new._get_full_pars(pars_log)
mr_new = new.cc.dim_dlog10rho0(cc_pars)



# numdifftools on old Jacobian
import numdifftools as nd
print 'pars_log', pars_log

print 'old'
Jac_func = lambda x: old.forward(x).T.reshape(frequencies.size * 2)
Jnum_old = nd.Jacobian(Jac_func, order=4)
J_old = old.Jacobian(pars_log)
J_old_num = Jnum_old(pars_log)
print 'num shape', J_old.shape, J_old_num.shape
diff_old = J_old - J_old_num
print 'Numdiff', diff_old.min(), diff_old.max()

# print J_old[3, :] / J_old_num[3, :]

print('new')
Jac_func = lambda x: new.forward(x).T.reshape(frequencies.size * 2)
Jnum_new = nd.Jacobian(Jac_func, order=4)
J_new_num = Jnum_new(pars_log)
print 'num shape', J_new_num.shape
diff_new = J_new - J_new_num
print 'Numdiff', diff_new.min(), diff_new.max()

# for line, linenum in zip(J_new, J_new_num):
#     d = line - linenum
#     print d.min(), d.max()

diffs_of_nums = J_old_num - J_new_num
print 'diffs_of_nums', diffs_of_nums

diffs_of_Js = J_old - J_new
for line in diffs_of_Js:
    print 'line', line.min(), line.max(), line[0]

