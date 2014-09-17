import lib_cc2.cc as CC
import numpy as np


class multi_colecole():

    def __init__(self, frequencies, nr_terms):
        self.nr_terms = nr_terms
        self.frequencies = frequencies

        self.cc = CC.colecole(frequencies)

    def set_parameters(self, pars):
        self.cc.set_parameters(pars)

    def imag(self):
        result = self.cc.imag()
        return result


class warburg():
    """Jury rig a warbug decomposition
    """
    def __init__(self, frequencies, tau):
        self.frequencies = frequencies
        self.tau = tau
        self.mcc = multi_colecole(frequencies, tau.size)

    def set_parameters(self, pars):
        rho0 = pars[0]
        m = pars[1:]

        rho0 = np.ones(m.size) * rho0
        c = np.ones(m.size) * 0.5
        parameters = (rho0, m, self.tau, c)
        self.mcc.set_parameters(parameters)

    def imag(self):
        results = self.mcc.imag()
        return results
