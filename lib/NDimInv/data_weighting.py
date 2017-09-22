""" Collection of helper functions which generate weightings for various base
data sets
"""
import numpy as np


def get_weighting_re_vs_im(base_data, settings):
    r"""Return a vector of weighting factors that compensate the mean of real
    and imaginary parts:

    :math:`W_{im} = \frac{\overline{|re|}}{\overline{|im|}}`

    We assume that base_data is of shape (N, 2) with (N, 0) the real parts, (N,
    1) the imaginary parts
    """
    work_data = base_data.flatten(order='F')

    center = int(work_data.size / 2)
    re = work_data[0: center]
    im = work_data[center:]

    re_mean = np.mean(np.abs(re))
    im_mean = np.mean(np.abs(im))

    factor_im = re_mean / im_mean
    errors = np.ones(work_data.shape)
    errors[center:] *= factor_im

    errors = errors.reshape(base_data.shape, order='F')
    return errors


def get_weighting_im_to_avg_re(base_data, settings):
    r"""Weigh imaginary parts to the mean of the real part

    :math:`W_{im} = \frac{\overline{|re|}}{\overline{|im|}}`

    We assume that base_data is of shape (N, 2) with (N, 0) the real parts, (N,
    1) the imaginary parts
    """
    work_data = base_data.flatten(order='F')

    center = int(work_data.size / 2)
    re = work_data[0: center]
    im = work_data[center:]

    re_mean = np.mean(np.abs(re))

    factor_im = re_mean / im
    errors = np.ones(work_data.shape)
    errors[center:] *= factor_im

    errors = errors.reshape(base_data.shape, order='F')
    return errors


def get_weighting_im_to_avg_re_error(base_data, settings):
    r"""Weigh imaginary parts to the mean of the real part

    :math:`W_{im} = \frac{\overline{|re|}}{\overline{|im|}}`

    We assume that base_data is of shape (N, 2) with (N, 0) the real parts, (N,
    1) the imaginary parts
    """
    work_data = base_data.flatten(order='F')

    center = int(work_data.size / 2)
    re = work_data[0: center]
    im = work_data[center:]

    threshold = 0.1
    pha = np.abs(np.arctan2(im, re)) * 1000
    indices = np.where(pha > threshold)[0]
    if indices.size == 0:
        raise Exception(
            'Data weighting is not possible, all phase values ' +
            'below {0} mrad'.format(threshold)
        )

    re_mean = np.mean(np.abs(re))

    factor_im = re_mean / im[indices]
    errors = np.ones(work_data.shape)

    errors[center + indices] *= factor_im

    errors = errors.reshape(base_data.shape, order='F')
    return errors


def get_weighting_all_to_one(base_data, settings):
    """
    Return a weighting vector which 'normalizes' all data points to one
    """
    errors = 1.0 / base_data
    return errors


def get_weighting_rel_abs(base_data):
    # data are in log
    # data = 10 ** base_data
    data = base_data
    errors = 0.03 * data + 0.01
    return 1.0 / errors


def get_weighting_one(base_data, settings):
    return np.ones(base_data.shape)


functions = {
    're_vs_im': get_weighting_re_vs_im,
    'avg_im': get_weighting_im_to_avg_re,
    'avg_im_err': get_weighting_im_to_avg_re_error,
    'one': get_weighting_one,
    'all_to_one': get_weighting_all_to_one,
    'rel_abs_error': get_weighting_rel_abs,
}
