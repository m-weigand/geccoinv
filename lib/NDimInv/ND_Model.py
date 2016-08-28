"""
Copyright 2014 Maximilian Weigand

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


class ND_Model(object):
    """
    For the model side we need to know some things about the underlying
    model. Here we use the Debye-Decomposition routines
    """
    def __init__(self, model, Data, extra_dims):
        self.extra_dims = extra_dims
        self.Data = Data
        self.obj = model
        # This function is used by the Iteration() class to plot spectra
        self.custom_plot_func = None

        """"
        Based upon the data base information, set the model base information:
        - determine tau-values (need frequencies for that)
        - determine starting parameters (need data for that)
        """
        self.M_base_dims = self.obj.get_model_base_dimensions()
        self.compute_starting_parameters()

        """
        for each dimension there can be multiple regularizations
        """
        # key == dimension
        self.regularizations = {}
        self.cached_Wms = {}

        # store function pointer for a steplength-selection function here
        self.steplength_selector = None

    def DM_iterator(self):
        """
        Generator for base_dimensional chunks of D and M, i.e. iterate over all
        extra dimensions of both D and M.

        Use as:

        >>> for d, m in self.DM_iterator():
        >>>     self.Data.D[d]
        >>>     self.M[m]

        This function provides the slices in the same order as an oder='F'
        argument for .flatten().
        """
        # TODO: should this be something like iteritems ?
        all_indices = [range(0, x[1][1]) for x in self.extra_dims.items()]
        extra_indices = itertools.product(*all_indices)

        # create slices for the base dimensions, corresponds to a list of ':'
        # for each base dimension in M or D
        M_base_indices = [slice(0, x[1][1]) for x in self.M_base_dims.items()]
        D_base_indices = [slice(0, x[1][1]) for x in
                          self.Data.D_base_dims.items()]

        # compute and save the forward responses
        for index in extra_indices:
            sm = tuple(M_base_indices + list(index))
            sd = tuple(D_base_indices + list(index))
            yield sd, sm

    def add_regularization(self, dimension, reg_object, lambda_object):
        """
        Parameters
        ----------
        dimension : number of dimension the new regularization belongs to
                    (starting with 0)
        reg_object : regularization object
        lam_method : method to choose the regularization parameter lambda for
                     this regularization from (fixed|quotient)
        lam_parameters : parameters for the lambda-selection process (e.g.
                         fixed lambda value)
        """
        if(dimension not in self.regularizations):
            self.regularizations[dimension] = []

        reg_set = (reg_object, lambda_object)
        self.regularizations[dimension].append(reg_set)

    def _get_steps(self, Msize):
        """
        TODO
        """
        steps = []
        for tmp_dim in range(0, len(Msize)):
            step = 1
            for i in range(0, tmp_dim):
                step *= Msize[i]
            steps.append(step)
        return steps

    def _permutate_indices(self, indices):
        if(not indices):
            return [0, ]

        dimind = indices[0]
        for item in indices[1:]:
            temp_ind = []
            for nr in item:
                for old_index in dimind:
                    if(type(old_index) == int):
                        old_index = [old_index, ]
                    temp_ind.append((old_index + [nr, ]))

            dimind = temp_ind
        return dimind

    def _get_offset(self, coordinates, dim_order, steps, dim):
        right_dims = []
        left_dims = []
        for nr2, i in enumerate(dim_order[1:]):
            if(i < dim):
                left_dims.append(coordinates[nr2])
            if(i > dim):
                right_dims.append(coordinates[nr2] * steps[i])
        offset = int(sum(right_dims))
        return offset

    def map_reg_matrix_to_global_Wm(self, dim, func, outside_first_dim):
        """
        Assemble the regularization matrix of one regularization function of
        the N dimensional problem. Depending on the function pointer that is
        provided, this can be either :math:`W_m^T \cdot W_m` or just
        :math:`W_m`

        TODO: Clean up, describe the procedure in more detail

        Parameters
        ----------
        dim : dimension to be regularized (int, starting with 0)
        func : function pointer for the regularization function, will be called
               as func(parametersize). This is usually a .Wm() function or
               WtWm() function.
        outside_first_dim : TODO
        """
        Msize = np.array(self.get_M_dimensions())

        # find all dimensions other than the requested one
        indices = list(set(range(0, Msize.size)) - set([dim, ]))

        # get new shape if we would swap the first with the regularized
        # dimension a'la Y = M.swapaxes(0, dim)
        new_size = Msize.copy()
        new_size[0], new_size[dim] = new_size[dim], new_size[0]

        # store changed order
        dim_order = range(0, len(Msize))
        dim_order[0] = dim
        dim_order[dim] = 0

        new_2d_shape = (np.prod(Msize[indices]), Msize[dim])
        Wm_small = func(new_2d_shape[1])
        size_x = new_2d_shape[0] * new_2d_shape[1]
        Wm_large = np.zeros((size_x, size_x))

        # # compute steps for each dimension
        # a step is the number of entries in the flattened version of the
        # parameter array which we have to skip in order to get the "neighbour"
        # of this regularization dimension
        steps = self._get_steps(Msize)

        indices = []
        for i in new_size[1:]:
            indices.append(range(0, i))

        # permutate indices
        dimind = self._permutate_indices(indices)

        last_offset = None  # store last offset to reset suboffset timer
        suboffset = -1

        # we collapse all dimensions expect for the regularization dimension,
        # and now have to get the regularization matrix for each of those
        # entries and map those to the global regularization matrix
        for nr in range(0, new_2d_shape[0]):
            width = new_2d_shape[1]
            # get coordinates of this iteration
            coordinates = dimind[nr]
            if(type(coordinates) == int):
                coordinates = [coordinates, ]

            if(dim > 0 and outside_first_dim is not None):
                if(coordinates[0] not in outside_first_dim):
                    continue

            # assign steps to the remaining dimensions, in the current order
            offset = 1

            offset = self._get_offset(coordinates, dim_order, steps, dim)
            if(last_offset is None):
                last_offset = offset

            if(last_offset != offset):
                last_offset = offset
                suboffset = -1

            suboffset += 1

            # 1) distance between entries: step
            # 2) offset: nr * pset
            start_x = offset + suboffset
            end_x = start_x + (width * steps[dim])

            # while the size of Wm is always the number of parameters, it can
            # vary for the y direction, depending on the type of matrix we are
            # aggregating. If we aggregate WtWm, we expect a NxN matrix. If we
            # aggregate only W, we expect an N-1xN matrix
            x = slice(start_x, end_x, steps[dim])
            y = self._get_y_slice(start_x, end_x, Wm_small.shape, dim, steps)
            Wm_large[y, x] = Wm_small
        return Wm_large

    def _get_y_slice(self, start_x, end_x, Wm_small_shape, dim, steps):
        nr_x, nr_y = Wm_small_shape
        if(nr_y == (nr_x - 1)):
            offset_y_slice = steps[dim]
        elif(nr_y == nr_x):
            offset_y_slice = 0
        else:
            print('Unexpected size of Wm!', Wm_small_shape)
            exit()

        y = slice(start_x, end_x - offset_y_slice, steps[dim])
        return y

    def retrieve_lams_and_WtWms(self, iteration):
        """
        Return a list of regularization matrices with corresponding lambda
        values.

        Parameters
        ----------
        iteration : the last iteration for which the new model update will be
                    computed
        """
        lambdas = []
        WtWms = []
        index = 0  # we need to know the absolute position of the lambda

        if(self.Data.extra_mask is None):
            dimensions = self.regularizations.keys()
        else:
            dimensions = [0, ]  # only base dimension

        for dimension in dimensions:
            for reg_set in self.regularizations[dimension]:
                # get the regularization matrix
                WtWm = self.map_reg_matrix_to_global_Wm(
                    dimension, func=reg_set[0].WtWm,
                    outside_first_dim=reg_set[0].outside_first_dim)
                # get the corresponding lambda value
                lambda_object = reg_set[1]
                lam = lambda_object.get_lambda(iteration, WtWm,
                                               lam_index=index)

                WtWms.append(WtWm)
                lambdas.append(lam)
                index += 1
        return lambdas, WtWms

    def get_M_dimensions(self):
        """
        get numbers of values for each dimension
        """
        sizes = [x[1][1] for x in self.M_base_dims.iteritems()]

        if(self.Data.extra_mask is None):
            sizes += [x[1][1] for x in self.extra_dims.iteritems()]
        return sizes

    def convert_to_M(self, m):
        """
        Given a flat vector m, return the multi-dimensional vecotor M shaped to
        the verious dimensions

        Parameters
        ----------
        m : flattened array
        """
        sizes = self.get_M_dimensions()
        M = m.reshape(sizes, order='F')
        return M

    def compute_starting_parameters(self):
        """
        Compute starting parameters for all spectra in the Data object
        """
        m0_list = []
        for d_slice, m_slice in self.DM_iterator():
            m0 = self.obj.estimate_starting_parameters(self.Data.D[d_slice])
            m0_list.append(m0)

        self.m0 = np.array(m0_list).flatten()

    def f(self, m):
        r"""
        Compute the forward response for the model parameters m

        Return the flattened response for ALL spectra in the form:

        :math:`\underline{f} = \begin{pmatrix}\underline{f}_1\\
        \underline{f}_2\\ \vdots\\ \underline{f}_n \end{pmatrix}`
        """
        response_list = []
        # nr of parameters (nr of tau values + 1 for rho0)
        step_size = self.M_base_dims[0][1]

        # loop over the spectra of all dimensions
        for index in range(0, m.size, step_size):
            pars = m[index: index + step_size]
            forward = self.obj.forward(pars).flatten(order='F')
            response_list.append(forward)
        response = np.array(response_list).flatten()
        return response

    def F(self, M):
        r"""Return the model responses in the same dimensionality as D, i.e.
        [base_dimensions, extra_dimensions].
        """
        F = np.zeros_like(self.Data.D)
        for d_slice, m_slice in self.DM_iterator():
            m_tiny = M[m_slice]
            F[d_slice] = self.obj.forward(m_tiny)
        return F

    def J(self, m):
        """
        Compute Jacobian for all dimensions
        """
        J_list = []
        # nr of parameters (nr of tau values + 1 for rho0)
        step_size = self.M_base_dims[0][1]
        for index in range(0, m.size, step_size):
            pars = m[index: index + step_size]
            J_temp = self.obj.Jacobian(pars)

            J_list.append(J_temp)

        # now build the block matrix
        block_height = J_list[0].shape[0]
        block_width = J_list[0].shape[1]

        matrix_width = len(J_list) * block_width
        matrix_height = len(J_list) * block_height

        J = np.zeros((matrix_height, matrix_width))

        for index in range(0, len(J_list)):
            start_x = index * block_width
            end_x = (index + 1) * block_width
            start_y = (index * block_height)
            end_y = (index + 1) * block_height
            J[start_y:end_y, start_x:end_x] = J_list[index]

        return J
