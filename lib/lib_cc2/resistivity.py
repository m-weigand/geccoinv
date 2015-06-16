"""
Implementation of the N-term Cole-Cole model in resistivities

TODO
----

* use some kind of closure or decorator to make sure we only call
  set_parameters once per overall call

"""
import numpy as np


class cc_res():

    def __init__(self, frequencies):
        self.rho0 = None
        self.m = None
        self.tau = None
        self.c = None
        self.f = frequencies
        # angular frequency
        self.w_orig = 2 * np.pi * self.f

    def set_parameters(self, pars):
        """Set the parameters for the whole class. Store as linear values.

        Parameters
        ----------

        pars: MxN array, where M is the number of spectra, and N = 1 + (p * 3)
              the number of parameters per spectrum, p the number of terms

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

        pars = np.atleast_2d(pars)

        nr_pars = pars.shape[1]
        self.rho0 = 10 ** (pars[:, 0])
        self.m = pars[:, 1:nr_pars:3]
        self.tau = 10 ** (pars[:, 2:nr_pars:3])
        self.c = pars[:, 3:nr_pars:3]

        # print 'pars'
        # print 'rho0', self.rho0
        # print 'm', self.m, self.m.shape
        # print 'tau', self.tau
        # print 'c', self.c

        # in order to compute multiple parameter sets at once (i.e. for a
        # decomposition approach), we need to resize the variables to enable
        # broadcasting
        new_size = (self.rho0.size, self.m.shape[1], self.w_orig.size)
        new_size_reversed = [x for x in reversed(new_size)]
        new_size_rho0 = (self.rho0.size, self.w_orig.size)
        new_size_rho0_rev = (self.w_orig.size, self.rho0.size)

        self.rho0 = np.resize(self.rho0, new_size_rho0_rev).T
        # print 'new rho0 shape', self.rho0.shape

        self.w = np.resize(self.w_orig, new_size)
        self.m = np.resize(self.m, new_size_reversed).T
        self.tau = np.resize(self.tau, new_size_reversed).T
        self.c = np.resize(self.c, new_size_reversed).T

        # print 'w.shape', self.w.shape
        # print 'm.shape', self.m.shape
        # print 'tau.shape', self.tau.shape

        # compute some common terms
        self.otc = (self.w * self.tau)**self.c
        self.otc2 = (self.w * self.tau)**(2 * self.c)
        self.ang = self.c * np.pi / 2  # rad
        self.denom = 1 + 2 * self.otc * np.cos(self.ang) + self.otc2

    def complex(self, pars):
        r"""Complex response:
        :math:`\hat{\rho} = \rho_0 \left(1 - \sum_i m_i (1 - \frac{1}{1 + (j
        \omega \tau_i)^c_i})\right)`
        """
        print 'complex pars', pars
        self.set_parameters(pars)
        terms = 1 - self.m * (1 - (1 / (1 + (1j * self.w * self.tau)**self.c)))
        # sum up terms
        specs = np.sum(terms, axis=1)
        rho = self.rho0 * specs

        return rho

    def mag(self, pars):
        r"""Magnitude :math:`|\hat{\rho}|`
        """
        self.set_parameters(pars)
        rho_complex = self.complex()
        mag = np.abs(rho_complex)
        return mag

    def pha(self, pars):
        r"""Phase shift (arctan2(im, rho)) [mrad]
        """
        self.set_parameters(pars)
        pha = 1000 * np.arctan2(self.imag(), self.real())
        return pha

    def mag_pha(self, pars):
        rho_complex = self.complex(pars)
        mag = np.abs(rho_complex)
        pha = 1000 * np.arctan2(np.imag(rho_complex), np.real(rho_complex))
        return mag, pha

    def real(self, pars):
        r"""
        :math:`\rho'(\omega) = \rho_0 \cdot \left(1 - m \frac{ (\omega \tau)^{c}
        \left(cos(\frac{c \pi}{2}) + (\omega \tau)^{c}\right)}{1 + 2 (\omega
        \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}}\right)`
        """
        rhoi = self.complex(pars)
        real = np.real(rhoi)
        # self.set_parameters(pars)
        # # helper terms
        # numerator = np.cos(self.ang) + self.otc
        # term = numerator / self.denom

        # real_cc = self.rho0 * (1 - self.m * self.otc * (term))
        return real

    def imag(self, pars):
        r"""Imaginary part
        :math:`\rho''(\omega) = m \frac{ - \rho_0 (\omega \tau)^{c}
        sin(\frac{c \pi}{2})}{1 + 2 (\omega
        \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}}`
        """
        rhoi = self.complex(pars)
        imag = np.imag(rhoi)
        # numerator = - self.rho0 * self.m * self.otc * np.sin(self.ang)
        # term = numerator / self.denom

        # imag_cc = term
        return imag

    def re_im(self, pars):
        r"""Return :math:`\hat{\rho}' and \hat{\rho}''`
        """
        re = self.real()
        im = self.imag()
        return re, im

    def dre_drho0(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho'}(\omega)}{\partial \rho_0} = 1 -
        \frac{m (\omega \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^c}{1 + 2
        (\omega \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}}`
        """
        self.set_parameters(pars)
        nominator = self.m * self.otc * np.cos(self.ang) + self.otc2
        term = nominator / self.denom
        specs = np.sum(term, axis=1)

        result = 1 - specs
        return result

    def dre_dlnrho0(self, pars):
        result = np.log(10) * self.rho0 * self.dre_drho0(pars)
        return result

    def dre_dm(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho'}(\omega)}{\partial m} = - \rho_0 m (\omega \tau)^c
        \frac{(cos(\frac{c \pi}{2}) + (\omega \tau)^c)}{1 + 2
        (\omega \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}}`
        """
        self.set_parameters(pars)
        nominator = -self.otc * np.cos(self.ang) + self.otc
        result = nominator / self.denom
        specs = np.sum(result, axis=1)
        specs *= self.rho0
        return specs

    def dre_dtau(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho'}(\omega)}{\partial \tau} = \rho_0
        \frac{-m \omega^c c \tau^{c-1} cos(\frac{c \pi}{2} - m \omega^{2 c} 2 c
        \tau^{2c - 1}}{1 + 2 (\omega \tau)^c cos(\frac{c \pi}{2}) + (\omega
        \tau)^{2 c}} +
        \rho_0 \frac{\left[m (\omega \tau)^c (cos(\frac{c \pi}{2}) + (\omega
        \tau)^c) \right] \cdot \left[ 2 \omega^c c \tau^{c-1} cos(\frac{c
        \pi}{2}) + 2 c \omega^{2 c} \tau^{2 c - 1}\right]}{\left[1 + 2 (\omega
        \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}\right]^2}`
        """
        self.set_parameters(pars)
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
        specs = np.sum(result, axis=1)
        specs *= self.rho0
        return specs

    def dre_dlntau(self, pars):
        # self.set_parameters(pars)
        result = 1.0 / self.real(pars) * self.dre_dtau(pars)
        return result

    def dre_dc(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho'}(\omega)}{\partial c} = \rho_0
        \frac{-m ln(\omega \tau) (\omega \tau)^c cos(\frac{c \pi}{2}) + m
        (\omega\tau)^c \frac{\pi}{2} sin(\frac{c \pi}{2}) + ln(\omega
        \tau)(\omega \tau)^c}{1 + 2 (\omega \tau)^c cos(\frac{c \pi}{2}) +
        (\omega \tau)^{2 c}} +
        \rho_0 \frac{\left[-m (\omega \tau)^c (cos(\frac{c \pi}{2}) + (\omega
        \tau)^c) \right] \cdot \left[ -2 ln(\omega \tau) (\omega \tau)^c
        cos(\frac{c \pi}{2}) + 2 (\omega \tau)^c \frac{\pi}{2} cos(\frac{c
        \pi}{2} + 2 ln(\omega \tau) (\omega \tau)^{2 c}\right]}{\left[1 + 2
        (\omega \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}\right]^2}`
        """
        self.set_parameters(pars)
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
        specs = np.sum(result, axis=1)
        specs *= self.rho0
        return specs

    def dim_drho0(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho}''(\omega)}{\partial \rho_0} = -
        \frac{m (\omega \tau)^c sin(\frac{c \pi}{2})}{1 + 2
        (\omega \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}}`
        """
        self.set_parameters(pars)
        nominator = - self.m * self.otc * np.sin(self.ang)
        terms = nominator / self.denom
        result = np.sum(terms, axis=1)
        return result

    def dim_dlnrho0(self, pars):
        result = 1 / self.imag(pars) * self.dim_drho0(pars)
        return result

    def dim_dm(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho''}(\omega)}{\partial m} = - \rho_0 m
        (\omega \tau)^c \frac{sin(\frac{c \pi}{2})}{1 + 2 (\omega \tau)^c
        cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}}`
        """
        self.set_parameters(pars)
        nominator = -self.otc * np.sin(self.ang)
        result = nominator / self.denom
        specs = np.sum(result, axis=1)
        specs *= self.rho0
        return specs

    def dim_dtau(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho''}(\omega)}{\partial \tau} = \rho_0
        \frac{-m \omega^c c \tau^{c-1} sin(\frac{c \pi}{2} }{1 + 2 (\omega
        \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}} +
        \rho_0 \frac{\left[-m (\omega \tau)^c sin(\frac{c \pi}{2}
        \right] \cdot \left[ 2 \omega^c c \tau^{c-1} cos(\frac{c
        \pi}{2}) + 2 c \omega^{2 c} \tau^{2 c - 1}\right]}{\left[1 + 2 (\omega
        \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}\right]^2}`
        """
        self.set_parameters(pars)
        # term1
        nom1 = - self.m * np.sin(self.ang) * self.w**self.c *\
            self.c * self.tau**(self.c - 1)
        term1 = nom1 / self.denom

        # term2
        nom2 = (- self.m * self. otc * np.sin(self.ang)) *\
            (2 * self.w**self.c * self.c * self.tau**(self.c - 1) *
             np.cos(self.ang) + 2 * self.c * self.w**(2 * self.c) *
             self.tau**(2 * self.c - 1))
        term2 = nom2 / self.denom**2

        result = term1 + term2
        specs = np.sum(result, axis=1)
        specs *= self.rho0
        return specs

    def dim_dlntau(self, pars):
        result = 1 / self.imag(pars) * self.dim_dtau(pars)
        return result

    def dim_dc(self, pars):
        r"""
        :math:`\frac{\partial \hat{\rho''}(\omega)}{\partial c} = \rho_0
        \frac{-m sin(\frac{c \pi}{2}) ln(\omgea \tau)(\omega \tau)^c - m
        (\omega \tau)^c \frac{\pi}{2} cos(\frac{\pi}{2}}{1 + 2 (\omega \tau)^c
        cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}} +
        \rho_0 \frac{\left[-m (\omega \tau)^c cos(\frac{c \pi}{2})
         \right] \cdot \left[ -2 ln(\omega \tau) (\omega \tau)^c
        cos(\frac{c \pi}{2}) + 2 (\omega \tau)^c \frac{\pi}{2} cos(\frac{c
        \pi}{2}) \right] + \left[2 ln(\omega \tau) (\omega \tau)^{2 c}\right]}{\left[1 + 2
        (\omega \tau)^c cos(\frac{c \pi}{2}) + (\omega \tau)^{2 c}\right]^2}`
        """
        self.set_parameters(pars)
        # term1
        nom1 = - self.m * np.sin(self.ang) *\
            np.log(self.w * self.tau) * self.otc - self.m *\
            self.otc * (np.pi / 2) * np.cos(self.ang)
        term1 = nom1 / self.denom

        # term2
        nom2 = (- self.m * self.otc * np.sin(self.ang)) *\
            ((- 2 * np.log(self.w * self.tau) * self.otc * np.cos(self.ang) +
             2 * self.otc * (np.pi / 2) * np.cos(self.ang)) +
             (2 * np.log(self.w * self.tau) * self.otc2))
        term2 = nom2 / self.denom**2
        result = term1 + term2

        specs = np.sum(result, axis=1)
        specs *= self.rho0
        return specs

    def Jacobian_re_im(self, pars):
        r"""
        :math:`J`
        """
        partials = []

        partials.append(self.dre_drho0(pars))
        partials.append(self.dre_dm(pars))
        partials.append(self.dre_dtau(pars))
        partials.append(self.dre_dc(pars))
        # partials.append(self.dim_drho0(pars))
        # partials.append(self.dim_dm(pars))
        # partials.append(self.dim_dtau(pars))
        # partials.append(self.dim_dc(pars))
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
