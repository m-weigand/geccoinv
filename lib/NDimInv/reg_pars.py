# *-* coding: utf-8 *-*
r"""
initial :math:`\lambda` value :math:`\lambda_0`
-----------------------------------------------

Multiple methods can be used to determine a suitable :math:`\lambda_0` value:

* (recommended) The mean of the diagonal entries of
  :math:`\underline{\underline{A}}^T \underline{\underline{W}}_d^T
  \underline{\underline{W}}_d \underline{\underline{A}}`

* *easylam*: The number of model parameters

* user determined

lambda selection for each iteration
-----------------------------------

* Fixed

* Factor/Quotient method: The lambda value of the last iteration is multiplied
  by 10 and 100; and divided by 10 and 100. The lambda values corresponding to
  the smallest resulting RMS-value is used for the next iteration.

Not implemented (yet):

* Golden section / Binary search

* Univariate search

* Line search

* L-curve (not implemented as an automatic :math:`\lambda` selector
"""
import logging
import sys
import itertools

import scipy.sparse as sparse
import numpy as np

import NDimInv.plot_helper
plt, mpl = NDimInv.plot_helper.setup()


class Lam0_Fixed(object):
    def __init__(self, fixed_lambda):
        self.value = fixed_lambda

    def get(self, it):
        return self.value


class Lam0_Easylam(object):
    def get(self, it):
        """
        A reasonable initial value for lambda is the number of parameters.
        """
        nr_of_parameters = it.Model.obj.get_data_base_size()
        return nr_of_parameters


class Lam0_AtWtWA(object):
    def get(self, it):
        """
        Determine the initial lambda value based on

        - Newman and Alaumbaugh, 1997
        - Kemna, 2000 (p.76)

        by using the mean value of the diagonal entries of A^T W_d^T D_d A
        """
        Wd = sparse.csc_matrix(self.Data.Wd)
        J = it.Model.J(it.m)
        J = sparse.csc_matrix(J)

        M = J.T.dot(Wd.T.dot(Wd.dot(J)))
        lam0 = np.mean(np.diag(M))
        return lam0


class BaseLambda(object):
    """
    Base class for lambda (regularization parameter) function. Here we
    implement the check for lambda 0

    """
    def get_lambda(self, it, WtWm, lam_index):
        """
        Parameters
        ----------
        it : iteration for which the model update will be computed
        WtWm : the regularization matrix for which the lambda will be computed
               for
        lam_index : the index in the iteration.lams list. Used to get the
                    lambda of the last iteration

        """
        if(it.nr == 0):
            lam = self.lam0_obj.get(it)
        else:
            old_lam = it.lams[lam_index]
            lam = self._get_lambda(it, WtWm, old_lam)
        return lam


class Lcurve(object):
    """ Create an L-curve for a given iteration.

    WARNING: This is not yet a finalized regulariazion object for use as an
    regularization object. For now it is used to plot the l-curve for the given
    iteration.
    """
    def __init__(self):
        # which rms key to use for optimization
        self.rms_key = 'rms_re_im_noerr'
        self.rms_index = 1

    def plot_lcurve(self, it, WtWms, lams, lam_index, output_prefix):
        """
        Parameters
        ----------


        Returns
        -------
        fig


        """
        rms_values, test_Rm, test_lams = self._sample_lambdas(it, WtWms, lams,
                                                              lam_index)
        if(test_lams == []):
            logging.info('No test lambdas available, returning')
            return

        fig, ax = plt.subplots(1, 1)
        ax.loglog(rms_values, test_Rm, '.-', color='k')
        index = 0
        for x, y in zip(rms_values, test_Rm):
            ax.annotate('{0:.4}'.format(float(test_lams[index])), xy=(x, y))
            index += 1

        ax.set_xlim([min(rms_values), max(rms_values)])
        ax.set_ylim([min(test_Rm), max(test_Rm)])
        ax.set_xlabel('RMS value: {0}'.format(self.rms_key.replace('_', '\_')))
        ax.set_ylabel(r'$\left| R \cdot m \right|$')
        ax.set_title('L-Curve based on iteration {0}'.format(it.nr))

        # save data to text files
        output_data = np.vstack((test_lams, test_Rm, rms_values)).T
        return fig, output_data

    def _sample_lambdas(self, it, WtWms, lams, lam_index):
        # these are the lambda values we will test
        test_lams_raw = np.logspace(-3, 3, 20)

        test_its = []
        test_lams = []
        test_Rm = []

        # here we store the lambda of all regularisation functions we use
        # we only change one (with index lam_index) for checks in the
        # l-curve
        lam_set = list(lams)

        # create iteration for each test lambda
        for test_lam in test_lams_raw:
            try:
                # try to create an iteration for this lambda
                test_it = it.copy()
                lam_set[lam_index] = test_lam
                # update_m = test_it._model_update((test_lam, ), (WtWm, ))
                update_m = test_it._model_update(lam_set, WtWms)

                # we could select an optimal alpha value here
                # alpha = self.Model.steplength_selector.get_steplength(
                #   self, update_m)
                alpha = 1
                test_it.m = it.m + alpha * update_m
                test_it.f = test_it.Model.f(test_it.m)
                Rm = np.sum(WtWms[lam_index].dot(test_it.m) ** 2)
                test_Rm.append(Rm)
                test_its.append(test_it)
                test_lams.append(test_lam)
            except Exception as e:
                logging.info(
                    'There was an error in the lambda test for ' +
                    'lambda: {0}. Trying next lambda.'.format(test_lam)
                )
                logging.info(e)
                continue

        # aggregate all RMS values
        rms_values = [x.rms_values[self.rms_key][self.rms_index]
                      for x in test_its]

        # remove nan values
        indices = np.where(np.isinf(rms_values))[0]
        for i in indices[::-1]:
            del(rms_values[i])
            del(test_Rm[i])
            del(test_lams[i])
        return rms_values, test_Rm, test_lams

    def _get_lambda(self, it, WtWm, lam_index):
        """
        this function call is compatible to the other regularization parameter
        selection functions, although not yet implemented to return an optimal
        lambda value.
        """
        # rms_values, test_Rm, test_lams = self._sample_lambdas(it, WtWm,
        #                                                      lam_index)
        # now find the optimal lambda values and return it
        return None


class SearchLambdaIndividual(BaseLambda):
    """
    This works only for dimension 0 regularizations!

    Test multiple lambda values for an optimal rms decrease

    Test for each spectrum individually and return a sparse diagonal matrix
    """
    def __init__(self, lam0_obj):
        """
        Parameters
        ----------
        lam0_obj : object which generates the first lambda values
        """
        self.lam0_obj = lam0_obj
        # which rms key to use for optimization
        self.rms_key = 'rms_re_im_noerr'
        self.rms_index = 1

    def _get_lambda(self, it, WtWm, old_lam):
        M = it.Model.convert_to_M(it.m)

        # the lambda search object we use to determine the individual lambdas
        SL = SearchLambda(self.lam0_obj)

        # extract the regularization of this spectrum from the large matrix
        len_m_small = it.Model.M_base_dims[0][1]
        WtWm_small = WtWm[0:len_m_small, 0:len_m_small]

        # loop through all spectra in the order we would flatten them
        # this is done by creating indices for all extra dimensions
        e_indices = [range(0, x[1][1]) for x in it.Data.extra_dims.items()]
        lambdas = []

        for nr, extra_index in enumerate(itertools.product(*e_indices)):
            # create an iteration with only this spectrum
            it_indiv = it.copy()
            it_indiv.Data.extra_mask = list(extra_index)
            # we assume that there is only one base dimension in M
            m_slice = [slice(0, M.shape[0]), ] + list(extra_index)
            it_indiv.m = M[tuple(m_slice)]
            it_indiv.f = it.Model.f(it_indiv.m)

            # # find optimal lambda for this iteration
            if(type(old_lam) is float or type(old_lam) is int):
                lam_old_individual = old_lam
            else:
                # extract from matrix
                # compute first index
                lam_index = len_m_small * nr
                lam_old_individual = old_lam[lam_index, lam_index]

            new_lam = SL._get_lambda(it_indiv, WtWm_small, lam_old_individual)

            # add N lambdas to list (N = number parameters for this spectrum)
            lambdas += it_indiv.m.size * [new_lam, ]
        # reset the extra_mask
        it.Data.extra_mask = None
        L = sparse.csc_matrix(np.diag(lambdas))
        return L


class SearchLambda(BaseLambda):
    """
    Test multiple lambda values for an optimal rms decrease
    """
    def __init__(self, lam0_obj, rms_key='rms_all_noerr', rms_index=0):
        """
        Parameters
        ----------
        lam0_obj : object which generates the first lambda values
        """
        self.lam0_obj = lam0_obj
        # which rms key to use for optimization
        self.rms_key = rms_key
        self.rms_index = rms_index

    def _get_lambda(self, it, WtWm, old_lam):
        """
        For the actual lambda value, compute model updates for lambda / 10;
        lambda * 10 and choose the version with the best rms

        Parameters
        ----------
        see BaseLambda.get_lambda(...)
        """
        lam_old = float(old_lam)

        # these are the lambda values we will test
        # for the frozen data sets we sometimes needed:
        # lam_old/1e7, lam_old/1e4
        # lam_old / 200, lam_old / 100, lam_old / 50,
        test_lams_raw = (lam_old / 10, lam_old / 5, lam_old * 5,
                         lam_old * 10, lam_old * 100, lam_old * 1e4)

        test_its = []
        test_lams = []

        # add initial iteration and old lambda
        test_its.append(it)
        test_lams.append(lam_old)

        # create iteration for each test lambda
        for test_lam in test_lams_raw:
            try:
                # try to create an iteration for this lambda
                test_it = it.copy()
                update_m = test_it._model_update((test_lam, ), (WtWm, ))
                # select optimal alpha value
                alpha = test_it.Model.steplength_selector.get_steplength(
                    test_it, update_m, True)
                # check if all steplengths lead to errors
                if(alpha is None):
                    continue
                # alpha = 1
                test_it.m = it.m + alpha * update_m
                test_it.f = test_it.Model.f(test_it.m)

                test_its.append(test_it)
                test_lams.append(test_lam)
            except:
                logging.info(
                    'There was an error in the lambda test for ' +
                    'lambda: {0}. Trying next lambda.'.format(test_lam)
                )
                e = sys.exc_info()[0]
                logging.info(e)
                continue

        # aggregate all RMS values
        rms_values = [x.rms_values[self.rms_key][self.rms_index]
                      for x in test_its]

        minimal_lambda_index = np.argmin(rms_values)
        best_lam = test_lams[minimal_lambda_index]
        logging.info('all lambdas {0}'.format(test_lams))
        logging.info('optimal lambda {0}'.format(best_lam))

        return best_lam


class FixedLambda(BaseLambda):
    """
    Implement various regularization parameters
    """
    def __init__(self, fixed_value):
        self.fixed_lambda = fixed_value

    def get_lambda(self, it, WtWm, lam_index):
        """
        Depending on a given Iteration, return a new regularization value
        """
        return self.fixed_lambda
