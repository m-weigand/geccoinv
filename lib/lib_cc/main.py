"""
Implement Cole-Cole related functions such as forward operator and Jacobian
See Kemna, 2000, Appendix C, p 166+


"""
import numpy as np


class cole_cole():
    """
    This class could serve as a base class for various CC parameterizations,
    but for now we only use this one
    """
    def __init__(self):
        self.data_format = "lnrmag_rpha"
        self.a_save = None
        self.a_pars = None

    def get_data_base_dimensions(self):
        """
        Return a dict with a description of the data base dimensions. In this
        case we have frequencies and re/im data
        """
        D_base_dims = {0: ['frequency', None],
                       1: ['rmag_rpha', 2]
                       }
        return D_base_dims

    def get_model_base_dimensions(self):
        """
        Return a dict with a description of the model base dimensions. In this
        case we have one dimension: the DD parameters (rho0, mi) where m_i
        denotes all chargeability values corresponding to the relaxation times.
        """
        M_base_dims = {0: ['rho0_m_tau_c', 4]}
        return M_base_dims

    def set_settings(self, settings):
        """
        Set the settings and call necessary functions
        """
        self.settings = settings

        # extract some variables
        self.frequencies = self.settings['frequencies']
        self.omega = 2.0 * np.pi * self.frequencies

    def estimate_starting_parameters_1(self, part1, part2):
        # we know we get part1 = lnrmag, part2 = pha
        return np.array([part1[0], 0.05, np.log(0.04), 0.6])

    def forward(self, pars):
        rmag_rpha = self.cc_lnrmag_rpha(pars)
        return rmag_rpha

    def cc_lnrmag_rpha(self, pars):
        rmag, rpha = self.cc_rmag_rpha(pars)
        return np.log(rmag), rpha

    def cc_rmag_rpha(self, pars):
        data_complex = self.cc_rcomplex(pars)
        data_real = np.real(data_complex)
        data_imag = np.imag(data_complex)
        rmag = np.abs(data_complex)

        # phases, [mrad]
        rpha = 1000 * np.arctan2(data_imag, data_real)

        return rmag, rpha

    def cc_rcomplex(self, pars):
        # determine number of Cole-Cole terms
        nr_cc_terms = (len(pars) - 1) / 3

        # extract the Cole-Cole parameters
        rho0 = np.exp(pars[0])
        m = pars[1:len(pars):3]
        tau = np.exp(pars[2:len(pars):3])
        c = pars[3:len(pars):3]

        # extract frequencies
        f = self.frequencies

        # prepare temporary array which will store the values of all CC-terms,
        # which later will be summed up
        term = np.zeros((f.shape[0], nr_cc_terms), dtype=np.complex128)

        # compute Cole-Cole function, each term separately
        for k in range(0, nr_cc_terms):
            term[:, k] = (m[k]) * (1 - 1 /
                                  (1 +
                                   ((0 + 1j) * 2 * np.pi * f * tau[k]) **
                                   c[k]))

        # sum up
        term_g = np.sum(term, 1)

        # multiply rho0
        Zfit = rho0 * (1 - term_g)

        return Zfit

    def Jacobian(self, pars):
        J = self.cc_jac_lnR_phi(pars)
        return J

    def cc_jac_lnR_phi(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho}}{\partial |\rho|}`
        """
        ret = np.vstack((self.cc_der_lnR(pars), self.cc_der_phi(pars)))
        return ret

    def cc_der_lnR(self, pars):
        rho = self.cc_rcomplex(pars)
        crho = rho.conjugate()

        factor = 1.0 / (2.0 * rho * crho)

        columns = []
        for func in (self.d_rho_d_lnrho0,
                     self.d_rho_d_m,
                     self.d_rho_d_lntau,
                     self.d_rho_d_c):
            derivative = func(pars)
            term1 = derivative.conjugate() * rho
            term2 = crho * derivative
            column = term1 + term2
            columns.append(column)

        results = np.array(columns)
        results *= factor

        return results.T

    def cc_der_phi(self, pars):
        rho = self.cc_rcomplex(pars)

        term2 = self.cc_drhodx_complex(pars) /\
            np.vstack((rho, rho, rho, rho)).T

        ret = 1j * (self.cc_der_lnR(pars) - term2)
        ret = np.real(ret)
        return ret

    def cc_drhodx_complex(self, pars):
        der_lnrho0 = self.cc_der_lnrho(pars)
        der_m = self.cc_der_m(pars)
        der_lntau = self.cc_der_lntau(pars)
        der_c = self.cc_der_c(pars)

        jac_complex = np.vstack((der_lnrho0, der_m, der_lntau, der_c))
        return jac_complex.T

    def cc_der_lnrho(self, pars):
        return self.cc_rcomplex(pars)

    def cc_der_m(self, pars):
        ret_complex = np.exp(pars[0]) *\
            (1 / (1 + (1j * self.frequencies * 2 * np.pi * np.exp(pars[2])) **
                  pars[3]) - 1)
        return ret_complex

    def cc_der_lntau(self, pars):
        a = - (np.exp(pars[0]) * pars[1] * (1j * self.frequencies * 2 *
                                            np.pi * np.exp(pars[2])) **
               pars[3]) / (1 + (1j * self.frequencies * 2 * np.pi *
                                np.exp(pars[2])) ** pars[3]) ** 2
        ret_komplex = a * pars[3]
        return ret_komplex

    def cc_der_c(self, pars):
        a = - (np.exp(pars[0]) * pars[1] *
               (1j * self.frequencies * 2 * np.pi * np.exp(pars[2])) **
               pars[3]) / (1 + (1j * self.frequencies * 2 * np.pi *
                                np.exp(pars[2])) ** pars[3]) ** 2
        ret_komplex = a * np.log(1j * self.frequencies * 2 * np.pi *
                                 np.exp(pars[2]))
        return ret_komplex

    def J_phi(self, pars):
        """
        Return the phase part of the Jacobian
        """
        results = []
        for func in (self.d_rho_d_lnrho0,
                     self.d_rho_d_m,
                     self.d_rho_d_lntau,
                     self.d_rho_d_c):
            results.append = func(pars)
        # TODO FINISH IMPLEMENTATION
        return results

    def d_rho_d_lnrho0(self, pars):
        r"""
        """
        derivative = self.cc_rcomplex(pars)
        return derivative

    def d_rho_d_m(self, pars):
        r"""
        return
        """
        rho0 = np.exp(pars[0])
        tau = np.exp(pars[2])
        c = pars[3]

        derivative = 1 / (1 + (1j * self.omega * tau) ** c)
        derivative -= 1
        derivative *= rho0
        return derivative

    def d_rho_d_lntau(self, pars):
        r""""
        return :math:`\frac{\partial \rho}{\partial ln(\tau)}`
        """
        a = self.a(pars)
        c = pars[3]
        derivative = a * c
        return derivative

    def d_rho_d_c(self, pars):
        r"""
        return :math:`\frac{\partial \rho}{\partial c}`
        """
        a = self.a(pars)
        tau = np.exp(pars[2])
        derivative = a * np.log(1j * self.omega * tau)
        return derivative

    def a(self, pars):
        r"""
        Return the variable a defined as
        :math:`\frac{-\rho_0 m (i \omega \tau)^c}{(1 + (i \omega \tau)^c)^2}`
        """
        if(self.a_save is None or
           self.a_pars is None or
           np.any(self.a_pars == pars)):
            rho0 = np.exp(pars[0])
            m = pars[1]
            tau = np.exp(pars[2])
            c = pars[3]

            a_nominator = - rho0 * m * (1j * self.omega * tau) ** c
            a_denominator = (1 + (1j * self.omega * tau) ** c) ** 2

            a = a_nominator / a_denominator
            self.a_save = a
            self.a_pars = pars
        return self.a_save


def get(parameterization):
    """
    Return an modelling object for the given parameterization
    """
    if(parameterization == 'logrho0_m_logtau_c'):
        return cole_cole()
    else:
        raise TypeError('Parameterization not known!')
