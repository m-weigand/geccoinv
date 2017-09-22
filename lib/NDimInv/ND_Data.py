""" Copyright 2014-2017 Maximilian Weigand

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import itertools
import NDimInv.data_weighting as data_weighting


class ND_Data(object):
    """ Data related functions
    """
    def __init__(self, model, extra_dims, data_weighting_key, settings):
        """
        Parameters
        ----------
        settings: dict
            global settings dict
        """
        self.D = None
        self.obj = model
        self.settings = settings

        self.data_converter = None
        if data_weighting_key not in data_weighting.functions:
            raise Exception('data weighting scheme "{0}" is not known!'.format(
                data_weighting_key
            ))
        self.data_weighting_func = data_weighting.functions[data_weighting_key]

        # #### model side ####
        self.D_base_dims = self.obj.get_data_base_dimensions()
        self.D_base_size = [x[1][1] for x in self.D_base_dims.items()]
        self.extra_dims = extra_dims

        self.dimensions = self.D_base_dims.copy()
        for key, item in self.extra_dims.iteritems():
            self.dimensions[key + len(self.D_base_size)] = item

        self._base_length = None

        # extra dimensional mask
        self.extra_mask = None

    def _check_position_with_dimensions(self, extra_position):
        """ Check if the provided position lies within the registered extra
        dimensional space

        Parameters
        ----------
        extra_position : list/tuple
            list of extra dimensions. If no extra dimensions are used, None has
            to be used.
        """
        # are there any extra dimensions?
        if(len(extra_position) == 0):
            if (len(self.extra_dims) == 0):
                return
            else:
                raise TypeError('We expect no extra dimensions')

        # correspond the number of extra dimensions?
        position_dims = len(extra_position) - 1
        if(position_dims not in self.extra_dims):
            raise TypeError(
                'The position requests more dimensions than were previously' +
                'registered')

        # are the requested indices located within the boundaries?
        for index, dimension in enumerate(extra_position):
            if(self.extra_dims[index][1] < (dimension + 1)):
                raise IOError('Requested extra dimensional position {0} ' +
                              'lies outside of registered size'.format(index))

    def D_iterator(self):
        """
        Generator for base_dimensional chunks of D, i.e. iterate over all
        extra dimensions of both D.

        Use as:

        >>> for d in self.D_iterator():
        >>>     self.Data.D[d]

        This function provides the slices in the same order as an oder='F'
        argument for .flatten().
        """
        all_indices = [range(0, x[1][1]) for x in self.extra_dims.iteritems()]
        extra_indices = itertools.product(*all_indices)

        # create slices for the base dimensions, corresponds to a list of ':'
        # for each base dimension in M or D
        D_base_indices = [slice(0, x[1][1]) for x in
                          self.D_base_dims.items()]

        # compute and save the forward responses
        for index in extra_indices:
            sd = tuple(D_base_indices + list(index))
            yield sd

    def WD(self):
        """ Assemble the data weighting matrix using one of the available data
        weighting functions
        """
        WD = np.zeros_like(self.D)
        for slice_d in self.D_iterator():
            weightings = self.data_weighting_func(
                self.D[slice_d],
                settings=self.settings,
            )
            WD[slice_d] = weightings
        return WD

    @property
    def Wd(self):
        errors = self.WD().flatten(order='F')
        Wd = np.diag(errors)
        return Wd

    @property
    def Df(self):
        """ Return a flattened version of D (data vector)

        Returns
        -------
        Df: numpy.ndarray
            flattened version of the data vector
        """
        D = self.D
        if(self.extra_mask is not None):
            index = [slice(0, x) for x in self.D_base_size] + self.extra_mask
            D = D[tuple(index)]

        Df = D.flatten(order='F')
        return Df

    @property
    def base_length(self):
        """ Return the length of the base dimensions in a flattened state
        """
        if(self._base_length is None):
            lengths = [x[1][1] for x in self.D_base_dims.iteritems()]
            base_length = 1
            for item in lengths:
                base_length *= item
            self._base_length = base_length

        return self._base_length

    def _allocate_D(self):
        """ Allocate the array for self.D according to the dimensions and
        numbers in self.D_base_dims and self.extra_dims

        This function must only be called once after all dimensions were added.
        """
        if(self.D is not None):
            return

        # determine new shape of D
        new_shape = []
        for dimension in self.D_base_dims.keys():
            new_shape.append(self.D_base_dims[dimension][1])

        for dimension in self.extra_dims.keys():
            new_shape.append(self.extra_dims[dimension][1])

        self.D = np.zeros(new_shape)

    def add_data(self, data, data_format, extra):
        """ Add data to the data space

        Parameters
        ----------
        data: np.ndarray
            corresponding to base dimensions
        data_format: string
            data format as found in sip_convert
        extra: tuple/list
            this tuple/list provides the position in the extra dimensional
            space for the data set. Use None if no extra dimensions are use.
        """
        if(self.data_converter is not None and
           self.obj.data_format is not None):
            data_converted = self.data_converter(
                data_format, self.obj.data_format, data)
        else:
            data_converted = data

        # allocate D if we didn't do it yet
        self._allocate_D()

        # check dimensions with self.D_extra_dims
        self._check_position_with_dimensions(extra)

        # update value nrs if appropriate
        for nr, dimension in enumerate(extra):
            new_dim = dimension + 1
            if(new_dim > self.extra_dims[nr][1]):
                self.extra_dims[nr][1] = new_dim

        # assign data
        # note: it is important to use a tuple to index a variable number of
        # dimensions
        # see:
        # file:///usr/share/doc/python-numpy-doc/html/user/basics.indexing.html
        index = [slice(0, x[1][1]) for x in self.D_base_dims.iteritems()]
        index += extra
        index = tuple(index)

        self.D[index] = data_converted
