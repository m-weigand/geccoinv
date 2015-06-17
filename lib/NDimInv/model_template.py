"""
Template class for models

Inherit from it when you implement fit models
"""
import abc


class model_template(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, settings):
        """
        Parameters
        ----------
        settings : dict containing the settings for the forward model. These
                   settings are mode dependent, but usually include a parameter
                   (e.g. 'x') which serves as the independent variable
                   associated with the model parameters.

        """
        self.settings = settings
        # set this to None if no data conversion is to done
        self.data_format = None

    @abc.abstractmethod
    def estimate_starting_parameters(self, base_data):
        """Given a data set of base data dimensions, return an initial guess
        (starting parameters) for the inversion.

        Parameters
        ----------
        base_data : input data with base dimensions

        Returns
        -------
        initial_pars : initial parameters, size model base dimensions

        """
        print('need a function ' +
              '"estimate_starting_parameters(self, base_data)"')
        exit()

    @abc.abstractmethod
    def forward(self, pars):
        """Return the forward response in base dimensions
        """
        print('"forward" function not implemented')
        exit()

    @abc.abstractmethod
    def Jacobian(self, pars):
        r"""Return the Jacobian corresponding to the forward response. The
        Jacobian has the dimensions :math:`B \times D \times M`

        TODO: Check the return dimensions
        """
        print('"Jacobian" function not implemented')
        exit()

    @abc.abstractmethod
    def get_data_base_size(self):
        """Usually you do not need to modify this.
        """
        size = sum([x[1][1] for x in
                    self.get_data_base_dimensions().iteritems()])
        return size

    @abc.abstractmethod
    def get_data_base_dimensions(self):
        """
        Returns
        -------
        Return a dict with a description of the data base dimensions. In this
        case we have frequencies and re/im data

        In the example down below, the frequencies will be set using the
        settings dict supplied to the __init__ function. Therefore the number of
        frequencies must be added dynamically here. Do not leave any values as
        None!
        """
        print('base data dimensions not set')
        exit()
        D_base_dims = {0: ['frequency', None],
                       1: ['rre_rmim', 2]
                       }
        return D_base_dims

    @abc.abstractmethod
    def get_model_base_dimensions(self):
        """Return a dict with a description of the model base dimensions. In
        this case we have one dimension: the DD parameters (rho0, mi) where m_i
        denotes all chargeability values corresponding to the relaxation times.
        """
        M_base_dims = {0: ['rho0_mi', self.tau.size + 1]}
        return M_base_dims

    @abc.abstractmethod
    def compute_par_stats(self, pars):
        r"""For a given parameter set (i.e. a fit result), compute relevant
        statistical values such as :math:`m_{tot}`, :math:`m_{tot}^n`,
        :math:`\tau_{50}`, :math:`\tau_{mean}`, :math:`\tau_{peak}`

        This is the way to compute any secondary results based on the fit
        results.

        Store in self.stat_pars = dict()
        """
        self.stat_pars = {}
        print('function "compute_par_stats" is not implemented')
        exit()
        return self.stat_pars
