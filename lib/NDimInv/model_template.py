"""
Template class for models
"""
import numpy as np


class model_template(object):

    def __init__(self):
        """

        """
        self.data_format = "rre_rmim"

    def estimate_starting_parameters(self, part1, part2):
        pass
        # return format?

    def set_settings(self, settings):
        """
        Set the settings and call necessary functions
        """
        self.settings = settings

        # extract some variables
        self.frequencies = self.settings['frequencies']
        self.omega = 2.0 * np.pi * self.frequencies

    def forward(self, pars):
        """
        Return the forward response as an flattened version of all base
        dimensions
        """
        pass

    def Jacobian(self, pars):
        pass

    def get_data_base_size(self):
        size = sum([x[1][1] for x in
                    self.get_data_base_dimensions().iteritems()])
        return size

    def get_data_base_dimensions(self):
        """
        Return a dict with a description of the data base dimensions. In this
        case we have frequencies and re/im data
        """
        D_base_dims = {0: ['frequency', None],
                       1: ['rre_rmim', 2]
                       }
        return D_base_dims

    def get_model_base_dimensions(self):
        """
        Return a dict with a description of the model base dimensions. In this
        case we have one dimension: the DD parameters (rho0, mi) where m_i
        denotes all chargeability values corresponding to the relaxation times.
        """
        M_base_dims = {0: ['rho0_mi', self.tau.size + 1]}
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
