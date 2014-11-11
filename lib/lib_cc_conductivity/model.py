"""
Implementation of the single-term Cole-Cole model using the conductivity
formulation.

This implementation uses only one polarisation term!

Notes:
    - we can easily implement caching in this class, i.e. for self.complex(...)
"""
import numpy as np


class colecole_conductivity(object):

    def __init__(self, frequencies):
        self.f = frequencies
        # angular frequency
        self.w = 2 * np.pi * self.f

    def complex(self, pars):
        r"""Complex response:

        :math:`\sigma(\omega) = \sigma_\infty \left(m \left[1 - \frac{1}{1 +
        (j \omega \tau)^c \right] \right)`
        Parameters
        ----------
        pars : [np.log(sigma0), m, np.log(tau), c]

        """
        sigma_infty = np.exp(pars[0]) / (1 - pars[1])
        sigma = sigma_infty * (1 - pars[1] /
                               (1 + (1j * self.w * np.exp(pars[2]))**pars[3]))
        return sigma

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
