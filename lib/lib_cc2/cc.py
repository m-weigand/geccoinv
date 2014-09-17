"""
Implementation of the single-term Cole-Cole model

Notes:
    - we can easily implement caching in this class, i.e. for self.complex(...)
"""
import numpy as np


class colecole():

    def __init__(self, frequencies):
        self.rho0 = None
        self.m = None
        self.tau = None
        self.c = None
        self.f = frequencies
        # angular frequency
        self.w = 2 * np.pi * self.f

    def set_parameters(self, pars):
        """Set the parameters for the whole class. Store as linear values.
        """
        # check sizes
        if(isinstance(pars[0], (long, int, float))):
            is_nr = [isinstance(i, (long, int, float)) for i in pars]
            if(not np.all(is_nr)):
                print('All parameters must have the same size!')
                return
        else:
            sizes = [len(i) for i in pars]
            diff = [i - sizes[0] for i in sizes]
            if(not np.all(np.where(diff == 0))):
                print('All parameters must have the same size!')
                return

        self.rho0 = np.exp(pars[0])
        self.m = pars[1]
        self.tau = np.exp(pars[2])
        self.c = pars[3]

        if(self.rho0.size > 1):
            # in order to compute multiple parameter sets at once (i.e. for a
            # decomposition approach), we need to resize the variables to enable
            # broadcasting
            new_size_reversed = (self.w.size, self.rho0.size)
            new_size = (self.rho0.size, self.w.size)

            self.w = np.resize(self.w, new_size)
            self.rho0 = np.resize(self.rho0, new_size_reversed).T
            self.m = np.resize(self.m, new_size_reversed).T
            self.tau = np.resize(self.tau, new_size_reversed).T
            self.c = np.resize(self.c, new_size_reversed).T

        # compute some common terms
        self.otc = (self.w * self.tau)**self.c
        self.otc2 = (self.w * self.tau)**(2 * self.c)
        self.ang = self.c * np.pi / 2  # rad
        self.denom = 1 + 2 * self.otc * np.cos(self.ang) + self.otc2

    def complex(self):
        r"""Complex response:
        :math:`\hat{\rho} = \rho_0 (1 - m(1 - \frac{1}{1 + (i \omega \tau)^c}))
        """
        rho = self.rho0 * (
            1 - self.m * (1 - (1 / (1 + (1j * self.w * self.tau)**self.c))))
        return rho

    def mag(self):
        r"""Magnitude (|rho|)
        """
        rho_complex = self.complex()
        mag = np.abs(rho_complex)
        return mag

    def pha(self):
        r"""Phase shift (arctan2(im, rho)) [mrad]
        """
        pha = 1000 * np.arctan2(self.imag(), self.real())
        return pha

    def mag_pha(self):
        mag = self.mag()
        pha = self.pha()
        return mag, pha

    def real(self):
        r"""Real part
        """
        # helper terms
        numerator = np.cos(self.ang) + self.otc
        term = numerator / self.denom

        real_cc = self.rho0 * (1 - self.m * self.otc * (term))
        return real_cc

    def imag(self):
        r"""Imaginary part
        """
        numerator = - self.rho0 * self.m * self.otc * np.sin(self.ang)
        term = numerator / self.denom

        imag_cc = term
        return imag_cc

    def re_im(self):
        re = self.real()
        im = self.imag()
        return re, im

    def dre_drho0(self):
        nominator = self.m * self.otc * np.cos(self.ang) + self.otc2
        result = 1 - nominator / self.denom
        return result

    def dre_dlnrho0(self):
        result = 1/self.real() * self.dre_drho0()
        return result

    def dre_dm(self):
        nominator = -self.rho0 * self.otc * np.cos(self.ang) + self.otc
        result = nominator / self.denom
        return result

    def dre_dtau(self):
        # term1
        nom1 = - self.m * self.w**self.c * self.tau**(self.c - 1) *\
            np.cos(self.ang) - self.m * self.w**(2 * self.c) *\
            2 * self.c * self.tau**(2 * self.c - 1)
        term1 = nom1 / self.denom

        # term2
        nom2 = self.m * self.otc * (np.cos(self.ang) + self.otc) *\
            (2 * self.w**self.c * self.c * self.tau**(self.c - 1) *
                np.cos(self.ang) + 2 * self.c * self.w**(2 * self.c) *
                self.tau**(2 * self.c - 1))
        term2 = nom2 / self.denom**2

        result = term1 + term2
        return result

    def dre_dlntau(self):
        result = 1 / self.real() * self.dre_dtau()
        return result

    def dre_dc(self):
        # term1
        nom1 = - self.m * np.log(self.w * self.tau) * self.otc *\
            np.cos(self.ang) + self.m * self.otc * (np.pi / 2) *\
            np.sin(self.ang) + np.log(self.w * self.tau) * self.otc
        term1 = nom1 / self.denom

        # term2
        nom2 = (- self.m * self.otc * (np.cos(self.ang + self.otc))) *\
            (- 2 * np.log(self.w * self.tau) * self.otc * np.cos(self.ang) +
             2 * self.otc * (np.pi / 2) * np.cos(self.ang) +
             2 * np.log(self.w * self.tau) * self.otc2)
        term2 = nom2 / self.denom**2

        result = term1 + term2
        return result

    def dim_drho0(self):
        nominator = - self.m * self.otc * np.sin(self.ang)
        result = nominator / self.denom
        return result

    def dim_dlnrho0(self):
        result = 1 / self.imag() * self.dim_drho0()
        return result

    def dim_dm(self):
        nominator = -self.rho0 * self.otc * np.sin(self.ang)
        result = nominator / self.denom
        return result

    def dim_dtau(self):
        # term1
        nom1 = - self.rho0 * self.m * np.sin(self.ang) * self.w**self.c *\
            self.c * self.tau**(self.c - 1)
        term1 = nom1 / self.denom

        # term2
        nom2 = (- self.rho0 * self.m * self. otc * np.sin(self.ang)) *\
            (2 * self.w**self.c * self.c * self.tau**(self.c - 1) *
             np.cos(self.ang) + 2 * self.c * self.w**(2 * self.c) *
             self.tau**(2 * self.c - 1))
        term2 = nom2 / self.denom**2

        result = term1 + term2
        return result

    def dim_dlntau(self):
        result = 1 / self.imag() * self.dim_dtau()
        return result

    def dim_dc(self):
        # term1
        nom1 = - self.rho0 * self.m * np.sin(self.ang) *\
            np.log(self.w * self.tau) * self.otc - self.rho0 * self.m *\
            self.otc * (np.pi / 2) * np.cos(self.ang)
        term1 = nom1 / self.denom

        # term2
        nom2 = (-self.rho0 * self.m * self.otc * np.sin(self.ang)) *\
            ((- 2 * np.log(self.w * self.tau) * self.otc * np.cos(self.ang) +
             2 * self.otc * (np.pi / 2) * np.cos(self.ang)) +
             (2 * np.log(self.w * self.tau) * self.otc2))
        term2 = nom2 / self.denom**2
        result = term1 + term2
        return result

    def Jacobian_re_im(self):
        """

        """
        partials = []

        partials.append(self.dre_dlnrho0())
        partials.append(self.dre_dm())
        partials.append(self.dre_dlntau())
        partials.append(self.dre_dc())
        partials.append(self.dim_dlnrho0())
        partials.append(self.dim_dm())
        partials.append(self.dim_dlntau())
        partials.append(self.dim_dc())
        J = np.array(partials)
        return J

    def dmag_drho0(self):
        pass

    def dmag_dm(self):
        pass

    def dmag_dtau(self):
        pass

    def dmag_dlntau(self):
        pass

    def dmag_dc(self):
        pass

    def dpha_drho0(self):
        pass

    def dpha_dm(self):
        pass

    def dpha_dtau(self):
        pass

    def dpha_dlntau(self):
        pass

    def dpha_dc(self):
        pass

    def Jacobian_mag_pha(self):
        pass
