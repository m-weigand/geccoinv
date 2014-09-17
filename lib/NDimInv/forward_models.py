import lib_dd.main as DDR
import lib_cc.main as CC

"""
Store available models in a dictionary

The key serves as a unique identifier for this model. Each entry is a tuple of
size 3 with the first entry containing the function to call in order to get an
object of this type, and the second entry contains the object parameters
explanation.

TODO:
    - maybe we should introduce some kind of setting dictionary for
      model-specific options such as number of tau values per frequency decade
"""
collection = {}
collection['dd_log10rho0log10m'] = (
    DDR.get, 'log10rho0log10m',
    'Debye Decomposition with rho0 and m regularized in log10')

collection['cc_logrho0_m_logtau_c'] = (
    CC.get, 'logrho0_m_logtau_c',
    'Cole-Cole model')


def model_infos():
    """
    Print model information
    """
    print('')
    print('----------------')
    print('Available models')
    print('[key]  -  [description]')
    for key in collection:
        print('Key: {0} - {1}'.format(key, collection[key][2]))


def get(model_name, settings):
    """
    Return the object created by calling the object function with the provided
    parameters
    """
    object_func = collection[model_name][0]
    obj = object_func(collection[model_name][1], settings)
    return obj
