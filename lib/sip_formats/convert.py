import numpy as np
"""
Convert between different representations for complex resistivity spectra

Basically we always have two parameters for each frequency. These two
parameters can be representated in various forms: conductivity/resistivity;
magnitude-phase/real-imaginary part.

Note that in this context it doesn't matter if we deal with conductivities or
conductances (same for resitivity and resistance).
"""


def split_data(data, squeeze=False):
    """
    Split 1D or 2D into two parts, using the last axis

    Parameters
    ----------
    data:
    squeeze : squeeze results to remove unnecessary dimensions
    """
    vdata = np.atleast_2d(data)
    nr_freqs = vdata.shape[1] / 2
    part1 = vdata[:, 0:nr_freqs]
    part2 = vdata[:, nr_freqs:]
    if(squeeze):
        part1 = part1.squeeze()
        part2 = part2.squeeze()
    return part1, part2


def to_complex(mag, pha):
    complex_nr = mag * np.exp(1j / 1000 * pha.astype(np.float32))
    return complex_nr


def generic_magpha_to_reim(mag, pha):
    """
    Generically convert magnitude and phase to real and imaginary part using
    the formula :math:`mag \cdot exp(1j / 1000 * pha)`

    Thus it is suitable for resistivities, multiply conductivity phases with -1
    """
    complex_nr = to_complex(mag, pha)
    real_part = np.real(complex_nr)
    imag_part = np.imag(complex_nr)
    return real_part, imag_part

####################
# # from converter ##
####################


def from_cre_cim(data):
    cre, cim = split_data(data)
    return cre, cim


def from_cre_cmim(data):
    cre, cmim = split_data(data)
    return cre, -cmim


def from_cmag_cpha(data):
    cmag, cpha = split_data(data)
    cre, cim = generic_magpha_to_reim(cmag, cpha)
    return cre, cim


def from_log10rmag_rpha(data):
    rlog10mag, rpha = split_data(data)
    Z = to_complex(10 ** rlog10mag, rpha)
    Y = 1 / Z
    real_part = np.real(Y)
    imag_part = np.imag(Y)
    return real_part, imag_part


def from_lnrmag_rpha(data):
    rlnmag, rpha = split_data(data)
    Z = to_complex(np.exp(rlnmag, rpha))
    Y = 1 / Z
    real_part = np.real(Y)
    imag_part = np.imag(Y)
    return real_part, imag_part


def from_rmag_rpha(data):
    rmag, rpha = split_data(data)
    Z = to_complex(rmag, rpha)
    Y = 1 / Z
    real_part = np.real(Y)
    imag_part = np.imag(Y)
    return real_part, imag_part


def from_rre_rmim(data):
    rre, rmim = split_data(data)
    Z = rre - 1j * rmim
    Y = 1 / Z
    real_part = np.real(Y)
    imag_part = np.imag(Y)
    return real_part, imag_part


def from_rre_rim(data):
    rre, rim = split_data(data)
    Z = rre + 1j * rim
    Y = 1 / Z
    real_part = np.real(Y)
    imag_part = np.imag(Y)
    return real_part, imag_part


##################
# # to converter ##
##################
# converts from conductiviy re/im to various formats

def to_cre_cim(cre, cim):
    data = np.hstack((cre, cim))
    return data


def to_cre_cmim(cre, cim):
    cmim = -np.array(cim)
    data = np.hstack((cre, cmim))
    return data


def to_cmag_cpha(cre, cim):
    Y = cre + 1j * cim
    cmag = np.abs(Y)
    cpha = np.arctan2(cim, cre) * 1000
    return np.hstack((cmag, cpha))


def to_rre_rim(cre, cim):
    Y = cre + 1j * cim
    Z = 1 / Y
    real_p = np.real(Z)
    imag_p = np.imag(Z)
    return np.hstack((real_p, imag_p))


def to_rre_rmim(cre, cim):
    Y = cre + 1j * cim
    Z = 1 / Y
    real_p = np.real(Z)
    mimag_p = -np.imag(Z)
    return np.hstack((real_p, mimag_p))


def to_rmag_rpha(cre, cim):
    Y = cre + 1j * cim
    Z = 1 / Y
    real_p = np.real(Z)
    imag_p = np.imag(Z)
    mag = np.abs(Z)
    pha = np.arctan2(imag_p, real_p) * 1000
    return np.hstack((mag, pha))


def to_log10rmag_rpha(cre, cim):
    rmag_rpha = to_rmag_rpha(cre, cim)
    mag_slice = slice(0, rmag_rpha.shape[1] / 2)
    log10rmag_rpha = rmag_rpha
    log10rmag_rpha[:, mag_slice] = np.log10(rmag_rpha[:, mag_slice])
    return log10rmag_rpha


def to_lnrmag_rpha(cre, cim):
    rmag_rpha = to_rmag_rpha(cre, cim)
    mag_slice = slice(0, rmag_rpha.shape[1] / 2)
    lnrmag_rpha = rmag_rpha
    lnrmag_rpha[:, mag_slice] = np.log(rmag_rpha[:, mag_slice])
    return lnrmag_rpha


# store the converter functions in dicts
from_converters = {
    'lnrmag_rpha': from_lnrmag_rpha,
    'log10rmag_rpha': from_log10rmag_rpha,
    'rmag_rpha': from_rmag_rpha,
    'rre_rim': from_rre_rim,
    'rre_rmim': from_rre_rmim,
    'cmag_cpha': from_cmag_cpha,
    'cre_cim': from_cre_cim,
    'cre_cmim': from_cre_cmim
}

to_converters = {
    'lnrmag_rpha': to_lnrmag_rpha,
    'log10rmag_rpha': to_log10rmag_rpha,
    'rmag_rpha': to_rmag_rpha,
    'rre_rim': to_rre_rim,
    'rre_rmim': to_rre_rmim,
    'cmag_cpha': to_cmag_cpha,
    'cre_cim': to_cre_cim,
    'cre_cmim': to_cre_cmim
}


def convert(input_format, output_format, data):
    """
    Convert from the given format to the requested format

    Parameters
    ----------
    input_format : format of input data (parameter 'data')
    output_format : format of output data
    data : numpy array containing data in specified input format

    Possible input/output formats:
        'lnrmag_rpha'
        'log10rmag_rpha'
        'rmag_rpha'
        'rre_rim'
        'rre_rmim'
        'cmag_cpha'
        'cre_cim'
        'cre_cmim'

    Internally we always convert to real part and imaginary part of
    conductivity, and then convert back to the output format.
    """
    if(input_format == output_format):
        return data

    if(input_format not in from_converters):
        raise KeyError('Input format {0} not known!'.format(input_format))

    if(output_format not in to_converters):
        raise KeyError('Output format {0} not known!'.format(output_format))

    cre, cim = from_converters[input_format](data)
    converted_data = to_converters[output_format](cre, cim)
    return converted_data
