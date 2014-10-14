"""
Forward model for the exponentation function
:math:`f(a, b, \underline{x}) = a \cdot exp(b * \underline{x})`
"""
import numpy as np


class exp_model(object):

    def __init__(self, settings):
        """

        """
        self.settings = settings
        self.data_format = "rre_rmim"

    def estimate_starting_parameters(self, part1, part2):
        return [1.0, 1.5]

    def set_settings(self, settings):
        """
        Set the settings and call necessary functions
        """
        self.settings = settings

    def forward(self, pars):
        """
        Return the forward response as a flattened version of all base
        dimensions
        """
        print 'PARS', pars
        y = pars[0] * np.exp(self.settings['x'] * pars[1])
        print 'Y', y
        return y

    def Jacobian(self, pars):
        x = self.settings['x']
        J = np.ones((self.settings['x'].size, 2))
        J[:, 0] = np.exp(pars[1] * x)
        J[:, 1] = pars[0] * x * np.exp(pars[1] * x)
        return J

    def get_data_base_size(self):
        size = sum([x[1][1] for x in
                    self.get_data_base_dimensions().iteritems()])
        return size

    def get_data_base_dimensions(self):
        """
        Return a dict with a description of the data base dimensions. In this
        case we have frequencies and re/im data
        """
        D_base_dims = {0: ['y', len(self.settings['x'])]}
        # 1: ['yim', len(self.settings['x'])]}
        return D_base_dims

    def get_model_base_dimensions(self):
        """
        Return a dict with a description of the model base dimensions.
        """
        M_base_dims = {0: ['pars', 2]}
        return M_base_dims

    def check_data(self, part1, part2):
        return True

    def compute_par_stats(self, pars):
        """
        For a given parameter set (i.e. a fit result), compute relevant
        statistical values such das :math:`m_{tot}`, :math:`m_{tot}^n`,
        :math:`\tau_{50}`, :math:`\tau_{mean}`, :math:`\tau_{peak}`

        Store in self.stat_pars = dict()
        """
        self.stat_pars = {}
        return self.stat_pars
