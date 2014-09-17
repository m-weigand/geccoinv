import numpy as np
import data_weighting
import sip_formats.convert as sip_converter


class ND_Data():
    """
    Data related functions
    """
    def __init__(self, model, extra_dims):
        self.D = None
        self.obj = model

        ##### model side ####
        self.D_base_dims = self.obj.get_data_base_dimensions()
        self.D_base_size = [x[1][1] for x in self.D_base_dims.items()]
        self.extra_dims = extra_dims
        self._base_length = None

        # extra dimensional mask
        self.extra_mask = None

    def _check_extra_size(self, extra):
        """
        Check if the requested position lies within the registered extra
        dimensional space
        """
        for index, dimension in enumerate(extra):
            if(self.extra_dims[index][1] < (dimension + 1)):
                raise IOError('Requested extra dimensional position {0} ' +
                              'lies outside of registered size'.format(index))

    @property
    def Wd(self):
        """
        Compute data weighting matrix. This matrix is solely dependent on the
        input data
        """
        errors = []
        dataset_size = self.obj.get_data_base_size()
        # loop through data sets
        # TODO: this should be implemented as an iterator
        for index in range(0, self.Df.size, dataset_size):
            basedata = self.Df[index:index + dataset_size]
            weightings = data_weighting.get_weighting_re_vs_im(basedata)
            #weightings = data_weighting.get_weighting_all_to_one(basedata)
            errors += list(weightings)

        errors = np.array(errors).flatten()
        Wd = np.diag(errors)
        return Wd

    @property
    def Df(self):
        """
        Return a flattened version of D (data vector)
        """
        D = self.D
        if(self.extra_mask is not None):
            index = [slice(0,  x) for x in self.D_base_size] + self.extra_mask
            D = D[tuple(index)]

        Df = D.flatten(order='F')
        return Df

    @property
    def base_length(self):
        """
        Return the length of the base dimensions in a flattened state
        """
        if(self._base_length is None):
            lengths = [x[1][1] for x in self.D_base_dims.iteritems()]
            base_length = 1
            for item in lengths:
                base_length *= item
            self._base_length = base_length

        return self._base_length

    def _allocate_D(self):
        """
        Allocate the array for self.D according to the dimensions and numbers
        in self.D_base_dims and self.extra_dims

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
        """
        Add data to the data space

        Parameters
        ----------
        data : ndarray corresponding to base dimensions
        data_format : data format as found in sip_convert
        extra : this tuple/list provides the position in the extra dimensional
                space for the data set. Use None if no extra dimensions are
                use.
        """
        data_converted = sip_converter.convert(data_format,
                                               self.obj.data_format, data)

        # allocate D if we didn't do it yet
        self._allocate_D()

        # check dimensions with self.D_extra_dims
        self._check_position_with_dimensions(extra)

        # check extra size
        self._check_extra_size(extra)

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

    def _check_position_with_dimensions(self, position):
        """
        Check if the provided position lie within the registered extra
        dimensional space

        Parameters
        ----------
        position : list/tuple of extra dimensions. If no extra dimensions are
                   used, None has to be used.
        """
        if(len(position) == 0):
            if (len(self.extra_dims) == 0):
                return
            else:
                raise TypeError('We expect no extra dimensions')

        position_dims = len(position) - 1
        if(position_dims not in self.extra_dims):
            raise TypeError(
                'The position requests more dimensions than were previously' +
                'registered')
