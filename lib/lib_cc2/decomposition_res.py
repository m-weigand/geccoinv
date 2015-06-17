"""Cole-Cole decomposition
"""
import numpy as np
import resistivity


class decomposition_resistivity():

    def __init__(self, settings):
        required_keys = ('frequencies', 'tau', 'c')
        for key in required_keys:
            if key not in settings:
                raise Exception('required key not found: {0}'.format(key))

        self.settings = settings
        self.cc = resistivity.cc_res(self.settings['frequencies'])

    def _get_full_pars(self, pars_dec):
        # prepare Cole-Cole parameters
        rho0 = pars_dec[0][np.newaxis]
        m = 10 ** pars_dec[1:]
        tau = np.log10(self.settings['tau'])
        if m.size != tau.size:
            raise Exception('m and tau have different sizes!')

        c = np.ones_like(m) * self.settings['c']

        pars = np.hstack((rho0, m, tau, c))
        return pars


    def forward(self, pars_dec):
        """

        Parameters
        ----------
        pars_dec: [rho0, m_i]

        """
        pars = self._get_full_pars(pars_dec)
        reim = self.cc.reim(pars)
        return reim

    def J(self, pars_dec):
        """
        Input parameters:
            log10(rho0)
            log10(m)
        """
        pars = self._get_full_pars(pars_dec)
        partials = []

        partials.append(self.cc.dre_dlog10rho0(pars)[:, np.newaxis, :])
        partials.append(self.cc.dre_dlog10m(pars))
        partials.append(self.cc.dim_dlog10rho0(pars)[:, np.newaxis, :])
        partials.append(self.cc.dim_dlog10m(pars))
        J = np.concatenate(partials, axis=1)
        return J

