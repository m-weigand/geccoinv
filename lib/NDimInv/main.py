""" Copyright 2014-2017 Maximilian Weigand

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# from memory_profiler import *
import os
import logging
import json

import scipy.sparse as sparse
import scipy.sparse.linalg as SL
import numpy as np

import ND_Model
import ND_Data
import NDimInv.plot_helper
plt, mpl = NDimInv.plot_helper.setup()
import NDimInv.reg_pars as reg_pars
import NDimInv.helper as helper
helper
log = logging.getLogger(__name__)


class RMS_control(object):
    """Manage RMS definitions and names. Inherited by NDimInv

    Register rms values in the dict self.rms_types. For each rms value, add an
    list containing bools for each dimension of D. True denotes dimensions that
    are summed up for the rms value, and False denotes dimensions, along which
    rms values will be computed for each entry. It is advisable to only set one
    dimension to False.
    Missing entries in the list are automatically filled with True-values,
    i.e. they are summed up. Thus, an empy list denotes the rms of all data
    values.

    Examples:

    >>> self.rms_types['rms_all'] = []
    >>> self.rms_types['rms_re_im'] = [True, False]
    """
    def __init__(self):
        # default rms over all data points
        self.rms_types = {'rms_all': []}
        self.rms_names = {'rms_all': ['rms_all', ]}

    def add_rms(self, key, definition, names):
        """register a new rms value

        Parameters
        ----------
        key: internal key for this rms
        definition: rms definition, see class documentation
        names: a list of names corresponding to the rms values. The list has
               either only one entry for all rms values (numbers will be added
               automatically), or the length must match the number of resulting
               rms values.
        """
        if key in self.rms_types:
            logging.info('WARNING: Duplicate RMS definition: {0}'.format(key))
        self.rms_types[key] = definition
        self.rms_names[key] = names

    def save_rms_definition(self, filename):
        # prepare a tuple containing both rms dicts
        rms_definition = (self.rms_types, self.rms_names)
        with open(filename, 'w') as fid:
            json.dump(rms_definition, fid)

    def load_rms_definition(self, filename):
        with open(filename, 'r') as fid:
            self.rms_types, self.rms_names = json.load(fid)


class RMS(object):
    """
    Augment the Iteration() class with RMS computation functions
    """

    @property
    def rms_values(self):
        r"""Compute RMS values based on the definitions in self.rms_values.
        Two RMS values are computed:

        .. math::

            RMS_{error} = \sqrt{\frac{1}{N} \sum_{i} \left( \frac{d_i -
                f_i}{\epsilon_i} \right)^2}

            RMS_{no~error} = \sqrt{\frac{1}{N} \sum_{i} \left(\frac{d_i -
                f_i}{1} \right)^2}

        RMS values without errors included are prepended with the string
        "_noerr", and with errors the string '_error' is added.
        """
        D = self.Data.D
        F = self.Model.F(self.Model.convert_to_M(self.m))
        diff = (D - F)

        WD = self.Data.WD()
        diff_err = diff * WD

        diff_sq = diff ** 2
        diff_err_sq = diff_err ** 2

        rms_values = {}
        for key, item in self.RMS.rms_types.iteritems():
            # determine which dimensions to sum up
            full_item = np.array(item + [True, ] * (len(D.shape) - len(item)))
            indices = np.where(full_item)[0]
            # remaining = np.where(~full_item)[0]

            for full_key, diff in zip((key + '_error', key + '_noerr'),
                                      (diff_err_sq, diff_sq)):
                rms_sum = np.sum(diff, axis=tuple(indices))
                N = np.prod([diff.shape[x] for x in indices])
                rms = np.atleast_1d(np.sqrt(rms_sum / N))
                rms_values[full_key] = rms
        return rms_values

    @property
    def old_rms_values(self):
        r"""
        Compute the rms values for the first part (usually real part or
        magnitude), the second part (usually imaginary part or phase), and both
        parts following the formula

        :math:`RMS=\sqrt{\frac{1}{N}\sum_i^N \frac{d_i - f_i(m)}{\epsilon_i}}`

        Returns
        -------
        rms_values : dict containing the following keys (
                     rms_part1_noerr, rms_part2_noerr, rms_both_noerr,
                     rms_part1_error, rms_part2_error, rms_both_error)

        """
        d = self.Data.Df
        f = self.Model.f(self.m)

        Wd = self.Data.Wd
        d_part1, d_part2 = self.split_data_into_parts(d)
        f_part1, f_part2 = self.split_data_into_parts(f)

        rms_values = []
        rms_values_ng = {}
        for di, fi, Wslice, name in (
            (d_part1, f_part1, slice(0, d_part1.size), 'part1'),
            (d_part2, f_part2, slice(d_part1.size, Wd.shape[0]), 'part2'),
                (d, f, slice(0, Wd.shape[0]), 'both')):
            diff = (di - fi)
            try:
                rms = np.sum(diff ** 2)
                rms /= float(di.size)
                rms = np.sqrt(rms)
            except FloatingPointError:
                rms = np.inf

            rms_values_ng['rms_' + name + '_noerr'] = rms

            try:
                rms_err = np.sum(Wd[Wslice, Wslice].dot(diff) ** 2)
                rms_err /= float(di.size)
                rms_err = np.sqrt(rms_err)
            except FloatingPointError:
                rms_err = np.inf

            rms_values_ng['rms_' + name + '_error'] = rms_err

            rms_values.append(rms)
        return rms_values_ng

    def split_data_into_parts(self, vector):
        """
        Split the provided vector into the two basic parts (re/im| mag/pha).
        This corresponds to a split in dimension 1
        """
        if(self.Data.extra_mask is None):
            Dsize = self.Data.D.shape
        else:
            Dsize = self.Data.D_base_size
        vec_temp = vector.reshape(Dsize, order='F')
        part1 = vec_temp[:, 0].flatten(order='F')
        part2 = vec_temp[:, 1].flatten(order='F')
        return part1, part2


class SearchSteplengthParFit(object):
    r"""
    Determine an optimal steplength parameter :math:`\alpha` by fitting a
    parabola through the results for :math:`\alpha \in [0, 0.5, 1]`.

    This procedure needs only two calculations of the model update
    (:math:`\alpha = [0.5, 1]`).

    By default we optimize the rms called 'rms_all_noerr'
    """
    def __init__(self, optimize='rms_all_noerr', optimize_index=0):
        """
        Parameters
        ----------
        optimize : which rms to optimize (default: all):
        """
        self.rms_key = optimize
        self.rms_index = optimize_index

    def get_steplength(self, it, par_update, ignore_all_err=False):
        """

        """

        old_rms = it.rms_values

        rms_values = []
        alpha_values = []

        rms_values.append(old_rms[self.rms_key][self.rms_index])
        alpha_values.append(0)

        for nr, test_alpha in enumerate((0.5, 1)):
            m_test = it.m + test_alpha * par_update
            it_test = it.copy()
            it_test.m = m_test
            try:
                test_rms = it_test.rms_values
                rms_values.append(test_rms[self.rms_key][self.rms_index])
                alpha_values.append(test_alpha)
            except FloatingPointError:
                pass
            except ArithmeticError:
                pass

        if(len(rms_values) != 3):
            logging.info(
                'Not all steplengths could be calculated. ' +
                'Cannot fit parabola'
            )
            return None

        # fit parabola
        x = np.array(alpha_values)
        y = np.array(rms_values)

        A = np.zeros((3, 3), dtype=np.float)
        A[:, 0] = x ** 2
        A[:, 1] = x
        A[:, 2] = 1
        a, b, c = np.linalg.solve(A, y)

        # compute minum minmum
        x_min = -b / (2 * a)

        # we need to make sure to lie in the range ]0, 1]
        if(x_min > 1):
            x_min = 1

        # use a default here, the inversion will probably end if we do not
        # improve the rms
        if(x_min <= 0):
            x_min = 0.1

        # debug plots
        if(False):
            fig, ax = plt.subplots(1, 1)
            ax.plot(x, y, '.')
            ax.set_xlabel('alpha')
            ax.set_label('rms')
            x_dense = np.linspace(0, 1, 30)
            ax.plot(x_dense, a * (x_dense ** 2) + b * x_dense + c, '-')
            ax.axvline(x_min)
            if('global_prefix' in it.Model.obj.settings):
                output_prefix = it.Model.obj.settings['global_prefix']
            else:
                output_prefix = ""
            filename = output_prefix + 'steplength_parbola_it{0}'.format(it.nr)
            fig.savefig(filename + '.png')
            fig.clf()
            plt.close(fig)
            del(fig)
        return x_min


class SearchSteplength(object):
    """
    Step length finder

    Check a list of fixed alpha values for the optimal value
    """
    def __init__(self, fixed_values=None, rms_key='rms_all_noerr',
                 rms_index=1):
        """
        Parameters
        ----------
        fixed_values : None for default values (0.1, 0.5, 1)
                       Provide a tuple of float values to set test values
        rms_key: RMS dict key to optimize
        rms_index: index for the provided RMS key
        """
        if(fixed_values is None):
            # 1e-5, 1e-4
            self.values = (1e-3, 1e-2, 0.1, 0.5, 1)
        else:
            self.values = fixed_values

        self.rms_key = rms_key
        self.rms_index = rms_index

    def get_steplength(self, it, par_update, ignore_all_err=False):
        r"""
        For a given Iteration and a new model update, find an optimal step
        length parameter :math:`\alpha`

        The variables self.rms_key/self.rms_index determine the RMS to use for
        the optimization.

        Note that we do not check if alpha == 0 yields optimal results, this is
        left to the stopping creteria.

        Return None if all steplengths lead to errors.

        :math:`m_{i+1} = m_i + \alpha \cdot \Delta m`

        Parameters
        ----------
        it : iteration object of last iteration
        par_update : model update for next iteration
        """
        old_rms = it.rms_values

        best_index = -1
        for nr, test_alpha in enumerate(self.values):
            m_test = it.m + test_alpha * par_update
            it_test = it.copy()
            it_test.m = m_test
            try:
                test_rms = it_test.rms_values

                if(best_index == -1):
                    best_index = nr
                    best_rms = test_rms[self.rms_key][self.rms_index]
                else:
                    if(old_rms[self.rms_key][self.rms_index] >
                       test_rms[self.rms_key][self.rms_index]):
                        best_index = nr
                        best_rms = test_rms[self.rms_key][self.rms_index]
            except FloatingPointError:
                pass
        if(best_index == -1):
            logging.info(
                'All tested steplengths caused exceptions. Somethings ' +
                'really wrong here'
            )
            return None

        logging.info('Found an optimal steplength ({0}) with rms {1} '.format(
            self.values[nr], best_rms) + '(old rms: {0})'.format(
            old_rms[self.rms_key][self.rms_index]))
        best_value = self.values[best_index]
        return best_value


class InversionControl(object):
    """
    This class augments the NDimInv class with inversion control functions. The
    actual inversion code can be found in the Inversion class, which augments
    the Iteration class.
    """
    def __init__(self):
        super(InversionControl, self).__init__()
        self.iterations = []
        # which rms to use for stopping criteria
        self.stop_rms_key = 'rms_all_noerr'
        self.stop_rms_index = 0

    def get_initial_iteration(self):
        """Return a bare iteration object initialized with the starting model
        and all registered RMS values.

        Usually only the first iteration is initialized using this function.
        Afterwards, new iterations are based upon the last iteration.
        """
        it = Iteration(0, self.Data, self.Model, self.RMS, self.settings)

        # set starting model
        it.m = self.Model.m0
        it.f = self.Model.f(it.m)

        # set initial regularisation (lam0) values
        it.lams, _ = self.Model.retrieve_lams_and_WtWms(it)
        return it

    def start_inversion(self):
        """
        Initialize the inversion
        """
        it0 = self.get_initial_iteration()
        self.iterations.append(it0)

        # we store any additional inversion settings here
        # leave this for later use
        self.inversion_settings = {}

    def run_inversion(self):
        """
        Convenience wrapper which initializes and runs the full inversion with
        all iterations. This function also checks stopping criteria
        """
        log.info('Running inversion')
        if(self.iterations == []):
            self.start_inversion()
        # self.iterations[-1].plot()
        stop_now = False
        while(stop_now is False and
              not self.stop_before_next_iteration() and
              self.iterations[-1].nr < self.settings['max_iterations']):
            logging.info('Iteration: {0}'.format(self.iterations[-1].nr + 1))

            new_iteration, stop_now = self.iterations[-1].next_iteration()

            # self.iterations[-1].plot()
            if(not stop_now):
                stop_now = self.check_stopping_criteria_before_update(
                    new_iteration)

            if(stop_now is False):
                self.iterations.append(new_iteration)

    def check_stopping_criteria_before_update(self, new_it):
        """
        Return True if one of the stopping criteria applies
        """

        # return if any NaN values are found in the new parameters
        if np.any(np.isnan(new_it.m)):
            return True

        # return if any value is below 1e-15
        if np.any(new_it.m[1:] < -15):
            return True
            raise Exception('values below 1e-15')

        rms_upd_eps = 1e-5  # min. requested rms change between iterations
        allowed_rms_im_increase_first_iteration = 1e2

        #
        nr = self.iterations[-1].nr
        old_rms = self.iterations[-1].rms_values[
            self.stop_rms_key][self.stop_rms_index]
        new_rms = new_it.rms_values[
            self.stop_rms_key][self.stop_rms_index]

        # if we are in the first iteration, then we allow a slight increase in
        # the imaginary RMS, but not above a certain threshold
        # TODO: Perhaps this threshold should be in RMS?
        if(new_rms > old_rms):
            if(nr == 0):
                increase = (new_rms - old_rms)
                if(increase > allowed_rms_im_increase_first_iteration):
                    logging.info(
                        'First iteration RMS-IM increase lies above: ' +
                        '{0}'.format(
                            allowed_rms_im_increase_first_iteration
                        )
                    )
                    return True
            else:
                # in all other cases: we do not allow an increase in rms
                logging.info('RMS Increase')
                return True

        # stop of the rms increase does not lie above a certain threshold
        rms_diff = np.abs(new_rms - old_rms)
        if(rms_diff < rms_upd_eps):
            logging.info('RMS update below threshold: {0} - {1} < {2}'.format(
                new_rms, old_rms, rms_upd_eps
            ))
            return True

        return False

    def run_n_iterations(self, n):
        """
        Compute n iterations

        No stopping criteria are evaluated.
        """
        for i in range(0, n):
            new_iteration, stop_now = self.iterations[-1].next_iteration()
            self.iterations.append(new_iteration)
            # self.iterations[-1].plot()

    def stop_before_next_iteration(self):
        """
        Check stopping criteria
        """
        # for now most of the stopping criteria are evaluated after the
        # computating of the model update, but before the application of this
        # model update to the previous iteration
        return False


class Inversion(RMS):
    """
    This class augments the Iteration class with the actual inversion functions
    """

    def __init__(self):
        super(Inversion, self).__init__()
        self._update_dict = {}

    def next_iteration(self):
        """
        Compute next iteration and return the model update

        Returns
        -------
        new_iteration : Iteration object with the next iteration
        stop_now : if something goes wrong with the next iteration, return
                   False here an the inversion is stopped
        """
        new_iteration = Iteration(self.nr + 1, self.Data, self.Model,
                                  self.RMS, self.settings)

        lams, WtWms = self.Model.retrieve_lams_and_WtWms(self)
        new_iteration.lams = lams
        # debug: plot l-curve
        # import reg_pars
        # l_curve = reg_pars.Lcurve()
        # space Wtwm
        # l_curve._get_lambda(self, WtWms[0], 0)
        # # debug end

        update_m = self._model_update(lams, WtWms)

        # its (memory-wise) expensive to keep all details of the inversion
        # process. Delete them if requested
        retain_it_details = os.getenv('NDIMINV_RETAIN_DETAILS')
        if retain_it_details != '1':
            self._update_dict.clear()

        alpha = self.Model.steplength_selector.get_steplength(self, update_m)
        self._update_dict['alpha'] = alpha
        if(alpha is None):
            return None, True
        new_iteration.m = self.m + alpha * update_m
        new_iteration.f = self.Model.f(new_iteration.m)
        return new_iteration, False

    def _add_regularisations(self, A, b, lams, WtWms):
        r"""
        Add the regularization terms :math:`\lambda_i
        \underline{\unerline{W}}_{m,i}` to the matrix A and to vector b.
        """
        # add regularizations
        for lam, WtWm in zip(lams, WtWms):
            WtWm_sparse = sparse.csc_matrix(WtWm)

            if(type(lam) is not int and not isinstance(lam, float)):
                A = A + lam.dot(WtWm_sparse)
                b = b - lam.dot(WtWm_sparse).dot(self.m)
            else:
                A = A + lam * WtWm_sparse
                b = b - lam * WtWm_sparse.dot(self.m)
        return A, b

    def _select_solver(self):
        """
        Based on the environment variable DD_SOLVER, return the function which
        solves the system of linear equations A*x + b using the call

        >>> x = solve_func(A, b)

        Possible choices for DD_SOLVER:

            *  std: use scipy.sparse.linalg.spsolve
            *  cg: use scipy.sparse.linalg.cg (sparse conjugate gradient)
        """
        if('DD_SOLVER' in os.environ):
            solver_id = os.environ['DD_SOLVER']
            if(solver_id == 'std'):
                solve_func = SL.spsolve
            elif(solver_id == 'cg'):
                solve_func = SL.cg
            else:
                logging.info('ERROR: Solver not known: {0}'.format(solver_id))
                exit()
        else:
            solve_func = SL.spsolve
        # print('Using function {0} to solve system of lin. equations'.format(
        #     solve_func))
        return solve_func

    def _prepare_mode_update(self, lams, WtWms):
        """

        """
        diff = self.Data.Df - self.f
        J = sparse.csc_matrix(self.Model.J(self.m))
        Wd = sparse.csc_matrix(self.Data.Wd)
        self._update_dict['diff'] = diff
        self._update_dict['J'] = J
        self._update_dict['Wd'] = Wd
        self._update_dict['A_without_reg'] = J.T.dot(Wd.T.dot(Wd.dot(J)))
        self._update_dict['b_without_reg'] = J.T.dot(Wd.T.dot(Wd.dot(diff)))
        A, b = self._add_regularisations(self._update_dict['A_without_reg'],
                                         self._update_dict['b_without_reg'],
                                         lams, WtWms)
        self._update_dict['A'] = A
        self._update_dict['b'] = b
        self._update_dict['lams'] = lams
        self._update_dict['WtWms'] = WtWms

    def _model_update(self, lams, WtWms):
        """
        Compute the actual model update with the regularisations provided

        Parameters
        ----------
        last_it : Iteration object on which this update is based on
        lams : list/tuple of lambdas
        WtWms : list/tuple of regularization matrices :math:`W^T \cdot W`

        Returns
        -------
        model_update : model update vector
        """
        self._prepare_mode_update(lams, WtWms)

        # gather components of update formula
        solve_func = self._select_solver()

        # solve the system of linear equations A x = b
        update_m = solve_func(self._update_dict['A'],
                              self._update_dict['b'])

        # save A and b to file
        # import pickle
        # with open('A.dat', 'wb') as outfile:
        #     pickle.dump(A, outfile, pickle.HIGHEST_PROTOCOL)
        # np.savetxt('b.dat', b)

        # sometimes we get a list, in this instance take only the first element
        # see documentation for the solver functions
        if(isinstance(update_m, tuple)):
            update_m = update_m[0]
        self._update_dict['update'] = update_m
        return update_m


class Iteration(Inversion):
    """ This class holds all information of one iteration
    """
    def __init__(self, nr, Data, Model, RMS, settings):
        """
        Parameters
        ----------
        nr : iteration nr
        Data : Data object
        Model : Model object
        rms_types : dict defining the rms types to be computed
        settings: settings-dict as used by the NDimInv object
        """
        super(Iteration, self).__init__()

        self.RMS = RMS
        self.nr = nr
        self.Data = Data
        self.Model = Model
        self.settings = settings

        # the following parameters will be computed during the inversion
        # process
        self.m = None
        self.f = None
        self.lams = None
        self.statpars = None

    def plot(self, filename=None, **kwargs):
        """Plot this iteration. Note that this function is basically a wrapper
        around either self._plot_default(), or, if set,
        self.Model.custom_plot_func.plot. This wrapper then takes care of
        saving the file to disc, but only if a filename was given.
        """
        fig = None
        if self.Model.custom_plot_func is None:
            fig = self._plot_default()
        else:
            fig = self.Model.custom_plot_func.plot(
                it=self, **kwargs
            )

        if filename is not None:
            output_filename = "plot_"
            if('global_prefix' in self.Model.obj.settings):
                output_filename += self.Model.obj.settings['global_prefix']
            else:
                pass

            output_filename += filename + '{0:04}.png'.format(self.nr)

            fig.savefig(output_filename, dpi=150)
            fig.clf()
            plt.close(fig)
        else:
            return fig

    def _plot_default(self):
        """ Default plot routine for complex resistivity data
        """
        response = self.Model.f(self.m)
        d = self.Data.Df
        nr_f = len(self.Data.obj.frequencies)
        nr_spectra = response.size / 2 / nr_f

        # split data
        slices = []
        resp = []
        for i in range(0, d.size, nr_f):
            slices.append(d[i:i + nr_f])
            resp.append(response[i:i + nr_f])

        size_y = 2 * nr_spectra
        fig, axes = plt.subplots(nr_spectra, 2, figsize=(7, size_y))
        axes = np.atleast_2d(axes)

        for nr, i in enumerate(range(0, nr_spectra * 2, 2)):
            # part 1
            ax = axes[nr, 0]
            ax.semilogx(self.Data.obj.frequencies, slices[i], '.-',
                        label='data')
            ax.semilogx(self.Data.obj.frequencies, resp[i], '.-', c='r',
                        label='fit')
            ax.legend(loc='best')

            # part 2
            ax = axes[nr, 1]
            ax.semilogx(self.Data.obj.frequencies, slices[i + 1], '.-')
            ax.semilogx(self.Data.obj.frequencies, resp[i + 1], '.-', c='r')

        fig.suptitle(self.nr)
        return fig

    def copy(self):
        """ Return a copy of this instance. This copy is not a full copy,
        self.Data/self.Model will only be copied by reference. Thus if you
        change them here they will be changed everywhere.
        """
        it_copy = Iteration(self.nr,
                            self.Data,
                            self.Model,
                            self.RMS,
                            self.settings)

        # copy all variables
        it_copy.m = self.m
        it_copy.lams = self.lams
        it_copy.f = self.f

        return it_copy

    @property
    def stat_pars(self):
        """ Aggregate statistical parameters for this iteration and return a
        dictionary
        """
        self.statpars = {}
        # loop over the m instances (i.e. the base dimensions)
        parsize = self.Model.M_base_dims[0][1]
        for nr, index in enumerate(range(0, self.m.size, parsize)):
            one_parset = self.m[index: index + parsize]
            single_par_stats = self.Model.obj.compute_par_stats(one_parset)
            # now sort into self.statpars
            for key, item in single_par_stats.iteritems():
                if(key not in self.statpars):
                    self.statpars[key] = []
                self.statpars[key].append(item)

        return self.statpars

    def plot_reg_strengths1(self, ax1, ax2):
        r"""
        Plot :math:`\left[\underline{\underline{J}}^T
        \underline{\underline{W}}_d^T \underline{\underline{W}}_d
        (\underline{d} - \underline{f}(\underline{m})) \right]^{-1}`
        """
        diff = self.Data.Df - self.f
        J = sparse.csc_matrix(self.Model.J(self.m))

        # compute update
        # base
        Wd = sparse.csc_matrix(self.Data.Wd)

        reg_strength1 = J.T.dot(Wd.T.dot(Wd.dot(diff)))
        # split into rho0 and chargeabilties
        size_m = self.Model.obj.tau.size

        rho0 = reg_strength1.copy()
        # mask chargeabilities
        for i in range(1, rho0.size, size_m + 1):
            rho0[i:i + size_m] = np.nan
        ax1.plot(rho0, '.-', color='k')
        ax1.set_xlim([0, len(reg_strength1)])
        ax1.grid(True)
        ax1.set_title('$rho_0$ - mean: {0} max: {1}'.format(
            np.mean(rho0), np.nanmax(rho0)))

        for i in range(0, rho0.size, size_m + 1):
            ax1.axvline(x=i, color='k', linestyle='dashed', linewidth=0.5)

        if('DD_USE_LATEX' in os.environ and
           os.environ['DD_USE_LATEX'] == '1'):
            ylabel_base = r'\left[\underline{\underline{J}}^T '
            ylabel_base += r'\underline{\underline{W}}_d^T '
            ylabel_base += r'\underline{\underline{W}}_d (\underline{d} - '
            ylabel_base += r'\underline{f}(\underline{m})) \right]^{-1}'
        else:
            ylabel_base = '[J^t W_d^T W_d (d - f(m))]^{-1}'
        ylabel_base = '$' + ylabel_base + '$'
        ax1.set_ylabel(ylabel_base)

        m = reg_strength1
        # mask rho0 values
        for i in range(0, m.size, size_m + 1):
            m[i] = np.nan

        ax2.plot(m, '.-', color='k')
        # now mark the start of each parameter set
        for i in range(0, m.size, size_m + 1):
            ax2.axvline(x=i, color='k', linestyle='dashed', linewidth=0.5)
        ax2.set_ylabel(ylabel_base)

        ax2.set_xlim([0, len(reg_strength1)])
        ax2.grid(True)
        ax2.set_title('$m_i$ - mean: {0} max: {1}'.format(
            m.mean(), np.nanmax(m)))

    # @profile
    def plot_reg_strengths(self, plot_to_file=False):
        r"""
        For each registered regularisation, plot :math:`\lambda_i \underline{
        \underline{W}}^T \underline{\underline{W}} \cdot \underline{m}` vs
        :math:`\underline{m}`
        """
        if('global_prefix' in self.Model.obj.settings):
            output_prefix = self.Model.obj.settings['global_prefix']
        else:
            output_prefix = ""

        outfile = output_prefix + 'reg_strength'
        lams, WtWms = self.Model.retrieve_lams_and_WtWms(self)
        fig, axes = plt.subplots(len(lams) + 2, 1,
                                 figsize=(7, len(lams) * 2.5 + 4))
        size_m = self.Model.obj.tau.size
        reg_index = 0
        for lam, WtWm in zip(lams, WtWms):
            ax = np.atleast_1d(axes)[reg_index]
            reg_strength = WtWm.dot(self.m)
            mean_strength = np.mean(reg_strength)
            if(isinstance(lam, (int, float))):
                reg_strength_lam = lam * reg_strength
                lam_value = lam
            else:
                reg_strength_lam = lam.dot(reg_strength)
                lam_value = 'indiv'

            ax.set_title('lam: {0}, mean wo lambda: {1}, max: {2}'.format(
                lam_value, mean_strength, np.max(reg_strength_lam)))
            ax.plot(reg_strength_lam, '.-', color='k')
            # now mark the start of each parameter set
            for i in range(0, self.m.size, size_m + 1):
                ax.axvline(x=i, color='k', linestyle='dashed', linewidth=0.5)
            ax.set_xlim([0, len(reg_strength)])
            ax.set_xlabel('Parameter number')
            if('DD_USE_LATEX' in os.environ and
               os.environ['DD_USE_LATEX'] == '1'):
                ylabel = r''.join((
                    r'$\lambda \cdot \underline{\underline{W}}^T ',
                    r'\underline{\underline{W}} \underline{m}$'
                ))
            else:
                ylabel = '$lam W^t W m$'

            ax.set_ylabel(ylabel)
            ax.grid(True)
            reg_index += 1

        self.plot_reg_strengths1(axes[-2], axes[-1])
        fig.tight_layout()

        if plot_to_file is True:
            fig.savefig(outfile + '.png')
            fig.clf()
            plt.close(fig)
            del(fig)

            # save plot data
            np.savetxt(outfile + '.dat', reg_strength_lam)
        else:
            return fig, reg_strength_lam

    def plot_lcurve(self, write_output=False):
        """ plot the L-curve after Hansen
        """

        logging.info('Plotting l-curve for iteration {0}'.format(self.nr))
        if('global_prefix' in self.Model.obj.settings):
            output_prefix = self.Model.obj.settings['global_prefix']
        else:
            output_prefix = ""

        LCURVE = reg_pars.Lcurve()
        lams, WtWms = self.Model.retrieve_lams_and_WtWms(self)

        figs = []
        outputs = []
        for lam_index in range(0, len(lams)):
            prefix = output_prefix + 'lam-nr_{0}_'.format(lam_index)
            fig, output = LCURVE.plot_lcurve(
                self, WtWms, lams, lam_index, prefix
            )
            figs.append(fig)
            outputs.append(output)
            if write_output:
                logging.info('saving lcruve to file')
                filename = prefix + 'l-curve-nr_{0}'.format(self.nr)
                fig.savefig(filename + '.png')
                plt.close(fig)

                # save data to text files
                header = '# lambda Rm RMS\n'
                with open(filename + '.dat', 'wb') as fid:
                    fid.write(
                        bytes(
                            header,
                            'utf-8',
                        )
                    )
                    np.savetxt(fid, output)

            lam_index += 1
        return figs, outputs


class NDimInv(InversionControl):
    """ N-dimensional model inversion for SIP-Spectra

    Data parameters
    ===============

    Data is stored in a numpy-array self.D

    Information on additional dimensions (extra dimensions) can be found in the
    dict self.D_extra_dims

    Model parameters are stored in self.M

    Usage examples
    ==============
    Implement the following use cases:

    1) Invert one spectrum (i.e. no extra dimensions)
    2) Time regularisation: invert 5 spectra for different time steps
    3) Spatial regularisation: invert 9 spectra for one time step at different
       locations
    4) Invert 9 x 5 spectra for 9 locations at 5 time steps

    TODO:
    =====
    * Find a way to check if all data was added or if we have 'white' spots in
      self.D

    """
    def __init__(self, model, settings):
        # call __init__ function of the InversionControl class
        super(NDimInv, self).__init__()

        self.Data = None
        self.Model = None
        self.settings = settings
        self._check_settings()
        self.model = model
        self.RMS = RMS_control()

        # #### extra dimensions #####
        # extra dimensions both are associated with the data and the model
        # space
        self.extra_dims = {}

    def _check_settings(self):
        """
        Check if all required settings were made
        """
        required_settings = ('max_iterations', )
        for requirement in required_settings:
            if(requirement not in self.settings):
                logging.info('NDimInv: Missing required setting {0}'.format(
                    requirement))
                exit()

    def finalize_dimensions(self):
        """ After all dimensions are added, create the Data space
        """
        data_weighting_function = self.settings.get(
            'data_weighting',
            None,
        )

        self.Data = ND_Data.ND_Data(
            self.model,
            self.extra_dims,
            data_weighting_function,
            self.settings,
        )

    def set_custom_plot_func(self, plot_obj):
        """
        Each iteration can plot the spectrum. We can provide a custom plot
        function to plot more detailed information
        """
        self.Model.custom_plot_func = plot_obj

    def update_model(self):
        """
        Create the model object
        """
        self.Model = ND_Model.ND_Model(self.model, self.Data,
                                       self.extra_dims)

    def _get_nr_dims(self, dim_dict):
        """
        Return number of dimensions for a provided dictionary

        Parameters
        ----------
        dim_dict: dict object such as self.D_base_dims or self.extra_dims
        """
        keys = dim_dict.keys()
        return len(keys)

    @property
    def nr_base_dims(self):
        """ Return number of base dimensions
        """
        return self._get_nr_dims(self.D_base_dims)

    @property
    def nr_extra_dims(self):
        """ Return number of extra dimensions
        """
        return self._get_nr_dims(self.extra_dims)

    def add_new_dimension(self, description, size):
        """ Add a new extra dimension to the inversion.

        Extra dimensions are added both to data and model space.

        Parameters
        ----------
        description : short string describing this dimension. Shouldn't contain
                      spaces
        size : number of entries on this dimension (e.g. how many time
                      steps or how many locations)
        """
        # get largest dimension number
        # use this as the index of the new dimension
        next_dim_nr = self.nr_extra_dims
        self.extra_dims[next_dim_nr] = [description, size]

    def _print_data_dimension_information(self, dim_dict):
        """ Print information regarding a dimension dict (e.g. base or extra)
        """
        for key in dim_dict.keys():
            logging.info(
                '{0} {1} [{2} values]'.format(
                    key, dim_dict[key][0],
                    dim_dict[key][1]
                )
            )

    def overview_data(self):
        """ Print information regarding the data space self.D
        """
        logging.info('\n\nDimension Information\n\n')
        logging.info('base dimensions')
        logging.info('---------------')
        self._print_data_dimension_information(self.Data.D_base_dims)
        if(self.Model is not None):
            self._print_data_dimension_information(self.Model.M_base_dims)

        logging.info('')
        logging.info('extra dimensions')
        logging.info('---------------')
        self._print_data_dimension_information(self.extra_dims)
