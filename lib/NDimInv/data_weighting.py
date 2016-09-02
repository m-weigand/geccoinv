"""
Collection of helper functions which generate weightings for various base data
sets
"""
import numpy as np


def get_weighting_re_vs_im(base_data):
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


def get_weighting_all_to_one(base_data):
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


def get_weighting_one(base_data):
    return np.ones(base_data.shape)


functions = {
    're_vs_im': get_weighting_re_vs_im,
    'one': get_weighting_one,
    'all_to_one': get_weighting_all_to_one,
    'rel_abs_error': get_weighting_rel_abs,
}
