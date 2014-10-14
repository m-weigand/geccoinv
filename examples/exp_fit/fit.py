#!/usr/bin/python
"""
Example of fitting an expontential function :math:`f(a, b, \underline{x}) = a
\cdot exp(b * \underline{x})`.
"""
import numpy as np
import model
import NDimInv
import NDimInv.regs as RegFuncs
import NDimInv.reg_pars as LamFuncs
from NDimInv.plot_helper import *


class plot_iteration(object):
    def plot(self, it, filename):
        fig, ax = plt.subplots(1, 1)
        ax.plot(it.Data.obj.settings['x'], it.Model.f(it.m), '-',
                label='iteration')
        ax.plot(it.Data.obj.settings['x'], it.Data.Df, '.',
                label='data')
        ax.legend()
        ax.set_title('a={0} b={1} rms={2}'.format(it.m[0], it.m[1],
                                                  it.rms_values['rms_all'][0]))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.savefig(filename)


if __name__ == '__main__':
    # generate data to be fitted
    x = np.arange(1, 2, 0.05)
    y = 2.0 * np.exp(x * 2.0)
    y += np.random.uniform(0, 5, x.size)

    # fit
    forward_settings = {'x': x}
    inversion_settings = {'max_iterations': 10}
    forward = model.exp_model(forward_settings)
    ND = NDimInv.NDimInv(forward, inversion_settings)
    ND.finalize_dimensions()
    ND.Data.add_data(y, None, extra=[])

    ND.update_model()
    ND.set_custom_plot_func(plot_iteration())

    lam_obj = LamFuncs.FixedLambda(1)

    ND.Model.add_regularization(0, RegFuncs.Damping(), lam_obj)

    optimize_for = 'rms_all'
    ND.Model.steplength_selector = NDimInv.main.SearchSteplengthParFit(
        optimize_for)

    ND.run_inversion()
    # ND.start_inversion()
    # ND.run_n_iterations(15)
    # ND.iterations[-1].plot()
    for it in ND.iterations:
        it.plot()
