#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This file contains some very old functions to handle grids used by CRTomo (see
Kemna, 2000): "Tomographic inversion of complex resistivity -- theory and
application".

GeccoInv (and dd_interfaces) was created to (among other applications) analyze
data produced by CRTomo. Thus this file.

Note that this file does not follow any coding standard, no OOP, and makes heavy
use of global variables. Do not use it for further development. It's only here
to make some functionality work until someone finds time to reimplement the
functionality in a proper way.

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
import numpy.ma as ma
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# grid variables:
global nodes, nodes_sorted
nodes = None
nodes_sorted = None

global grid_is_rectangular
grid_is_rectangular = True

# for triangle grids
global triangles
triangles = None

# for rectangular grids
global gridx, gridz
gridx = None
gridz = None


class cls_configs:
    def __init__(self):
        self.A = None
        self.B = None
        self.M = None
        self.N = None

configs = cls_configs()
global width_in_elements, height_in_elements
width_in_elements = -1
height_in_elements = -1

global electrode_pos_x, electrode_pox_z
electrode_pos_x = None
electrode_pos_z = None


class element:
    def __init__(self):
        self.nodes = np.empty(1)
        self.xcoords = np.empty(1)
        self.zcoords = np.empty(1)

global element_type_list
element_type_list = []

global filter_mask
filter_mask = None

"""
The following lists hold the grid data that will be plotted
"""
global element_data
element_data = None
global config
config = None

global node_data
node_data = None

"""
Plot options
"""


class plot_options:
    def __init__(self):
        self.cbmin = None
        self.cbmax = None
        self.grayscale = False
        self.reverse = False
        self.cblabel = ''
        self.title = ''
        self.title_position = None
        self.xlabel = 'x (cm)'
        self.ylabel = 'z (cm)'
        self.xmin = None
        self.xmax = None
        self.zmin = None
        self.zmax = None
        self.norm_fac = None
        self.dyn_fac = None
        self.cb_nr_tiks = None
        self.x_nr_tiks = None
        self.y_nr_tiks = None
        self.clip_electrodes = False
        self.no_elecs = False


global plt_opt
plt_opt = plot_options()

# global figure for plotting
global fig
fig = plt.figure()


def get_colormap():
    """
    Depending on the user settings, return a colormap
    """
    if(plt_opt.grayscale is False):
        if(plt_opt.reverse is True):
            cmap = mpl.cm.jet
        else:
            cmap = mpl.cm.jet_r
    else:
        if(plt_opt.reverse is True):
            cmap = mpl.cm.binary_r
        else:
            cmap = mpl.cm.binary
    return cmap


def plot_elements_to_file(output, cid, scale='linear'):
    """
    Plot element data to a grid and save to file
    """
    fig.clf()
    plot_element_data(cid, 111, scale)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


def plot_nodes_to_file(output, cid, fill_contours=True, levels=None):
    """
    Plot node data as a contour plot on a grid and save to file
    """
    fig.clf()
    plot_node_data(cid, 111, fill_contours, levels)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


def plot_element_data(cid, subplot_id, scale='linear'):
    """
    Plot element data to a grid. Only plot to the global figure,
    no output.
    """
    # frame_on controls the frame
    ax = fig.add_subplot(subplot_id, frame_on=False)
    if(plt_opt.no_elecs):
        no_elecs = True
    else:
        no_elecs = False
    plot_element_data_to_ax(cid, ax, scale, no_electrodes=no_elecs)
    return ax


def plot_node_data(cid, subplot_id, fill_contours=True, levels=None):
    """
    Plot node data to a grid as a contour plot. Only plot to the global figure,
    no output.
    """
    mpl.rcParams["font.size"] = 24.0

    x, y = get_dimensions_in_elements()
    X, Y = get_grid()

    xlim, zlim = _get_grid_limits()
    colors = np.reshape(node_data[:, cid], (y + 1, x + 1), order='C')

    # frame_on controls the frame
    ax = fig.add_subplot(subplot_id, frame_on=False)

    cmap = get_colormap()
    if(plt_opt.cbmin is None):
        plt_opt.cbmin = np.min(colors)
    if(plt_opt.cbmax is None):
        plt_opt.cbmax = np.max(colors)

    norm = mpl.colors.Normalize(vmin=float(plt_opt.cbmin),
                                vmax=float(plt_opt.cbmax))

    # levels for contourf
    if(levels is None):
        levels_neg = -np.logspace(-2, np.log10(np.abs(np.min(colors))), 14)
        levels_pos = np.logspace(-2, np.log10(np.max(colors)), 14)
        levels = np.sort(np.hstack((levels_neg, levels_pos)))

    if(fill_contours):
        pm = ax.contourf(X, Y, colors, levels=levels, cmap=cmap, norm=norm)
    else:
        pm = ax.contour(X, Y, colors, levels=levels, cmap=cmap, norm=norm)

    # version check
    if(int(mpl.__version__[0]) == 1):
        print('good version')
        # colorbar same hight as plot
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.12, pad=0.15)
        cb = plt.colorbar(pm, orientation='vertical', cax=cax)
    else:
        print('old version')
        cb = plt.colorbar(pm, orientation='vertical')

    # labels
    # cb.set_label(options.cblabel)

    ax.set_xlabel(plt_opt.xlabel)
    ax.set_ylabel(plt_opt.ylabel)
    ax.set_title(plt_opt.title)

    ax.titleOffsetTrans._t = (0., 10.0 / 72.0)
    # default ist (0., 5/72)
    ax.titleOffsetTrans.invalidate()

    # bad version check
    if(mpl.__version__[0] == 1):
        # fit axes to plot
        ax.autoscale(enable=True, axis='both', tight=True)
        # ax.autoscale_view()  # tight scaling to image extent

    # plot electrodes
    plot_electrodes_to_ax(ax)

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_aspect('equal')

    # run through all lines drawn for xticks and yticks
    for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
        line.set_visible(False)

    return ax, cb


def _get_grid_limits():
    """
    Depending on the user settings and the node coordinates, return the grid
    limits to use for plotting.
    """
    nodes = get_nodes()

    xlim = [0, 0]
    if(plt_opt.xmin is None or plt_opt.xmin == '*'):
        xlim[0] = np.min(nodes[:, 1])
    else:
        xlim[0] = float(plt_opt.xmin)

    if(plt_opt.xmax is None or plt_opt.xmax == '*'):
        xlim[1] = np.max(nodes[:, 1])
    else:
        xlim[1] = float(plt_opt.xmax)

    zlim = [0, 0]
    if(plt_opt.zmin is None or plt_opt.zmin == '*'):
        zlim[0] = np.min(nodes[:, 2])
    else:
        zlim[0] = float(plt_opt.zmin)

    if(plt_opt.zmax is None or plt_opt.zmax == '*'):
        zlim[1] = np.max(nodes[:, 2])
    else:
        zlim[1] = float(plt_opt.zmax)

    return xlim, zlim


def _get_colors(cid, scale):
    """
    Return the color values for the grid cells, and the limits (min, max).

    Also, transform the data according to the required scale.
    Possible scale values are:
    'linear' - do not transform the data at all
    'log' - log10 the data
    '10' - 10 ** the data
    'asinh' - perform a arcsinh transformation on the data. Consult the CRLab
              manual for detailed information (section about sens.dat).
    """
    if(grid_is_rectangular):
        x, y = get_dimensions_in_elements()
        colors = np.reshape(element_data[:, cid], (y, x), order='C')
    else:
        colors = np.squeeze(element_data[:, cid])

    if(scale == 'log' or scale == 'log10'):
        colors = np.log10(colors)
    elif(scale == '10'):
        colors = 10 ** colors
    elif(scale == 'asinh'):
        if(plt_opt.norm_fac is None):
            norm_fac = np.max(np.abs(colors))
        else:
            norm_fac = plt_opt.norm_fac

        if(plt_opt.dyn_fac is None):
            dyn_fac = np.abs(np.min(np.log10(np.abs(colors))))
        else:
            dyn_fac = plt_opt.dyn_fac

        colors = np.arcsinh(10 ** (dyn_fac) * colors / norm_fac) /\
            np.arcsinh(10 ** dyn_fac)

        if(plt_opt.cblabel == 'fill'):
            plt_opt.cblabel = 'norm = {0}'.format(norm_fac)
            plt_opt.title = r'$sinh^{-1}(10^{' + '{0:.3f}'.format(dyn_fac) +\
                            '} \cdot S_{norm}) / sinh^{-1}(10^{' +\
                            '{0:.3f}'.format(dyn_fac) + '})$'

    colors = ma.masked_where(np.isnan(colors), colors)
    colors = ma.masked_where(np.isinf(colors), colors)

    if(filter_mask is not None):
        print('Applying filter mask')
        if(grid_is_rectangular):
            colors = ma.masked_where(np.reshape(filter_mask, (y, x),
                                                order='C') == 0, colors)
        else:
            colors = ma.masked_where(filter_mask == 0, colors)

    if(plt_opt.cbmin is None):
        cbmin = np.min(colors)
    else:
        cbmin = plt_opt.cbmin

    if(plt_opt.cbmax is None):
        cbmax = np.max(colors)
    else:
        cbmax = plt_opt.cbmax

    if(cbmax < cbmin):
        cbmin, cbmax = cbmax, cbmin

    # if the data has only one value, adjust the cb-limits
    if(cbmin == cbmax):
        cbmin = cbmin - 0.2
        cbmax = cbmax + 0.2

    return colors, cbmin, cbmax


def plot_element_data_to_ax(cid, ax, scale='linear', no_cb=False,
                            no_electrodes=False, plot_gridlines=False):
    """
    Plot element to to a grid, given an axis object

    Parameters
    ----------
    cid : id for previously loaded data
    ax : axes object to plot to
    scale : [linear|log10] (see _get_colors for more options)
    no_cb : plot colorbar or not
    plot_gridlines : plot the grid
    """
    xlim, zlim = _get_grid_limits()

    colors, cbmin, cbmax = _get_colors(cid, scale)

    # we want to export those
    global cmap
    global norm
    cmap = get_colormap()
    norm = mpl.colors.Normalize(vmin=float(cbmin), vmax=float(cbmax))

    if(plot_gridlines):
        edgecolor = 'k'
    else:
        edgecolor = 'face'

    if(grid_is_rectangular):
        X, Y = get_grid()
        pm = ax.pcolormesh(X, Y, colors, edgecolor=edgecolor, linewidth=1e-2,
                           cmap=cmap, norm=norm, rasterized=True)
    else:
        tx = nodes_sorted[:, 1]
        ty = nodes_sorted[:, 2]
        # edgecolor='k' plots the grid lines
        pm = ax.tripcolor(tx, ty, triangles, edgecolor=edgecolor,
                          facecolors=colors, cmap=cmap, norm=norm)

    if(no_cb is False):
        # version check
        if(int(mpl.__version__[0]) == 1):
            # colorbar same hight as plot
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.12, pad=0.15)
            cb = plt.colorbar(pm, orientation='vertical', cax=cax)
        else:
            print('You are using an old version of Matplotlib ({0}).' +
                  'The colorbar can only be resized according to some crude' +
                  'heuristics!'.format(mpl.__version__))
            # determine X and Z dimensions
            width = np.abs(xlim[1] - xlim[0])
            height = np.abs(zlim[1] - zlim[0])
            if(height < width):
                ratio = height / width
            else:
                ratio = width / height

            cax, kw = mpl.colorbar.make_axes(ax, shrink=ratio)
            cb = plt.colorbar(pm, cax=cax, orientation='vertical')

        if(plt_opt.cb_nr_tiks is not None):
            cb.locator = mpl.ticker.MaxNLocator(nbins=plt_opt.cb_nr_tiks)
            cb.update_ticks()

        # labels
        cb.set_label(plt_opt.cblabel)
    else:
        cb = None
    ax.set_xlabel(plt_opt.xlabel)
    ax.set_ylabel(plt_opt.ylabel)

    if (plt_opt.title_position is None):
        ax.set_title(plt_opt.title)
    else:
        ax.set_title(plt_opt.title, position=plt_opt.title_position)

    # ax.title.set_y(1.05) # check way to set the title padding
    ax.titleOffsetTrans._t = (0., 10.0 / 72.0)
    # default ist (0., 5/72)
    ax.titleOffsetTrans.invalidate()

    # bad version check
    if(int(mpl.__version__[0]) == 1):
        # fit axes to plot
        ax.autoscale(enable=True, axis='both', tight=True)
        # ax.autoscale_view()  # tight scaling to image extent

    # plot electrodes
    if(not no_electrodes):
        plot_electrodes_to_ax(ax)

    # why do we need this?
    # fig.tight_layout()

    # run through all lines drawn for xticks and yticks
    for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
        line.set_visible(False)

    # set user supplied nr of x/y tick labels
    if(plt_opt.x_nr_tiks is not None):
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(
            nbins=plt_opt.x_nr_tiks))

    if(plt_opt.y_nr_tiks is not None):
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(
            nbins=plt_opt.y_nr_tiks))

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_aspect('equal')

    # ax.get_figure().canvas.draw()  # needed to get the labels/positions
    # # now we want to check for overlapping y tick labels
    # axy = ax.get_yaxis()
    # #print dir(axy)
    # bboxes = []
    # for i in axy.get_majorticklabels():
    #    #print i.get_label()
    #    #print i.get_text()
    #    #print dir(i)
    #    bboxes.append(i._bbox)
    # print bboxes

    return ax, pm, cb


def plot_electrodes_to_ax(ax, add_numbers=False):
    """
    Plot the electrodes on a given axes object.

    If add_numbers == True, then annotate each electrode with its number
    """
    Ex, Ez = get_electrodes()

    nr = 1
    for x, z in zip(Ex, Ez):
        if((plt_opt.xmin == '*' or plt_opt.xmin is None or
            x >= float(plt_opt.xmin)) and
           (plt_opt.xmax == '*' or plt_opt.xmax is None or
            x <= float(plt_opt.xmax)) and
           (plt_opt.zmin == '*' or plt_opt.zmin is None or
            z >= float(plt_opt.zmin)) and
           (plt_opt.zmax == '*' or plt_opt.zmax is None or
                z <= float(plt_opt.zmax))):
            ax.scatter(x, z, c='k', s=10, clip_on=plt_opt.clip_electrodes)

        if(add_numbers):
            ax.annotate('{0}'.format(nr), (x, z), fontsize=5.0,
                        textcoords='offset points', xytext=(3, 3))
        nr += 1


def plot_electrodes_to_ax_selected(ax, add_numbers=False, add_labels=None,
                                   selected_electrodes=None):
    """
    Plot the electrodes on a given axes object.

    If add_numbers == True, then annotate each electrode with its number
    """
    Ex, Ez = get_electrodes(selected_electrodes)

    nr = 1
    for x, z in zip(Ex, Ez):
        if((plt_opt.xmin == '*' or plt_opt.xmin is None or
            x >= float(plt_opt.xmin)) and
            (plt_opt.xmax == '*' or plt_opt.xmax is None or
             x <= float(plt_opt.xmax)) and
            (plt_opt.zmin == '*' or plt_opt.zmin is None or
             z >= float(plt_opt.zmin)) and
            (plt_opt.zmax == '*' or plt_opt.zmax is None or
                z <= float(plt_opt.zmax))):
            ax.scatter(x, z, c='k', s=10, clip_on=False)

        if(add_numbers):
            ax.annotate('{0}'.format(nr), (x, z), fontsize=5.0,
                        textcoords='offset points', xytext=(3, 3))
        nr += 1


def load_column_file_to_elements(filename, columns):
    """
    Import data to be plotted on the grid from a file with the following
    structure:

    1. Line: Number of entries
    2. - to [Number of entries + 1]: One or more columns of data points

    Parameters
    ----------
    filename : file to import
    columns : list with indices to load, starting with 0

    WARNING: This function assumes that the first line contains as many columns
    as the rest of the file!

    Return a list with the entries in the global data matrix, which were added

    """
    tmp_data = np.loadtxt(filename, dtype=float, skiprows=1)

    global element_data
    if(element_data is None):
        element_data = tmp_data[:, columns]
        indices = range(0, element_data.shape[1])
    else:
        indices = element_data.shape[1]
        element_data = np.append(element_data, tmp_data[:, columns], axis=1)
        indices = range(indices, element_data.shape[1])
    return indices


def load_column_file_to_elements_advanced(
    filename, columns, first_line_contains_nr_elements=True,
        no_header_line=False):
    """
    Import column data from a data file.

    Read out the first line to get the number of elements (rows).
    Otherwise behave as load_column_file_to_elements

    Parameters
    ----------

    filename: Data filename containing column data
    columns: List of column ids to import, starting from zero. Examples:
             [0,4,3] or [1,]

    first_line_contains_nr_elements: False - read as many elements a loaded in
                                     the grid. Exit if no grid was
                                     loaded yet!
                                     True  - read as many elements as written in
                                     the first line

    no_header_line: True: No header line is expected, the first line is assumed
                    to be data

    Returns
    -------
    indices: list of internal indices that identify the loaded columns. These
             indices can the be passed to the plotting functions.

    """

    fid = open(filename, 'r')

    if(first_line_contains_nr_elements is True):
        first_line = fid.readline().strip()
        first_line = np.fromstring(first_line, dtype=int, sep=' ')[0]
    else:
        first_line = len(element_type_list[0])
        if(no_header_line is False):
            fid.readline()
    tmp_data = np.zeros((first_line, len(columns)))
    for line_nr in range(0, first_line):
        line_str = fid.readline().strip()
        tmp_data[line_nr, :] = np.fromstring(line_str, dtype=float,
                                             sep=' ')[columns]

    indices = add_to_element_data(tmp_data)

    return indices


def add_to_element_data(subdata, fill_or_cut=False):
    """
    Add a new element data set to the global list of data sets. Return the
    index to this new data set.

    Parameters
    ----------
    element_data has two dimensions: [nr_of_cells, ids]
    fill_or_cut: if True, then fill with NaN values or cut to corresponding
                 element nr
    """
    # make sure we have a two dimension subdata
    if len(subdata.shape) == 1:
        subdata_2d = np.atleast_2d(subdata).T
    else:
        subdata_2d = subdata

    nr_elements = get_nr_of_elements()
    if fill_or_cut and subdata_2d.shape[0] > nr_elements:
        subdata_2d = subdata_2d[0:nr_elements, :]
    if fill_or_cut and subdata_2d.shape[0] < nr_elements:
        tmp = np.ones((nr_elements, subdata_2d.shape[1])) * np.nan
        tmp[0:subdata_2d.shape[0], :] = subdata_2d
        subdata_2d = tmp

    global element_data
    if element_data is None:
        element_data = subdata_2d
        indices = range(0, element_data.shape[1])
    else:
        indices = element_data.shape[1]
        element_data = np.append(element_data, subdata_2d, axis=1)
        indices = range(indices, element_data.shape[1])
    return indices


def load_column_file_to_nodes(filename, columns):
    """
    Important: For node files there is no first line with the number of
    elements
    """
    tmp_data = np.loadtxt(filename, dtype=float)
    tmp_data[:, 0:2] = np.round(tmp_data[:, 0:2], 5)

    tmp_data = resort_node_data(tmp_data)

    global node_data
    if(node_data is None):
        node_data = tmp_data[:, columns]
        indices = range(0, node_data.shape[1])
    else:
        indices = node_data.shape[1]
        node_data = np.append(node_data, tmp_data[:, columns], axis=1)
        indices = range(indices, node_data.shape[1])
    return indices


def convert_to_conductivity(mid, pid):
    """
    For given element ids, assume resistivity magnitude and phase. Then,
    convert to conductivity and return the conductivity magnitude and phase ids
    of the element_data array.

    We assume that the magnitudes are in log10!
    """
    global element_data

    # get data
    magnitudes = 10 ** element_data[:, mid]
    phases = element_data[:, pid]

    # convert
    resistivities = magnitudes * np.exp(1j / 1000 * phases)
    conductivities = 1 / resistivities
    mag_cond = np.abs(conductivities)
    pha_cond = np.arctan(np.imag(conductivities) / np.real(conductivities))
    # convert to mrad
    pha_cond *= 1000

    # save
    indices = [element_data.shape[1], element_data.shape[1] + 1]
    element_data = np.append(element_data, mag_cond, axis=1)
    element_data = np.append(element_data, pha_cond, axis=1)

    return indices


def convert_to_conductivity_re_im(mid, pid):
    """
    For given element ids, assume resistivity magnitude and phase. Then,
    convert to conductivity and return the conductivity magnitude and phase ids
    of the element_data array.

    We assume that the magnitudes are in log10!
    """
    global element_data

    # get data
    magnitudes = 10 ** element_data[:, mid]
    phases = element_data[:, pid]

    # convert
    impedances = magnitudes * np.exp(1j / 1000 * -phases)
    conductance = 1 / impedances
    real_cond = np.real(conductance)
    imag_cond = np.imag(conductance)

    # save
    indices = [element_data.shape[1], element_data.shape[1] + 1]
    element_data = np.append(element_data, real_cond, axis=1)
    element_data = np.append(element_data, imag_cond, axis=1)

    return indices


def resort_node_data(node_data):
    """
    If we read node data from a file, we have to take acount the changed
    positions due to CutMcK.
    """

    X, Y = get_grid()
    tpots = np.empty(X.shape)
    for rows in range(0, X.shape[0]):
        for cols in range(0, X.shape[1]):
            a = np.where(node_data[:, 0] == X[rows, cols])
            b = np.where(node_data[:, 1] == Y[rows, cols])
            c = np.intersect1d(a[0], b[0])
            # resort column index 2
            tpots[rows, cols] = node_data[c, 2]

    node_data[:, 2] = tpots.reshape(node_data.shape[0])
    return node_data


def load_pot_file(filename):
    """
    Wrapper for load_column_file_to_nodes to load a pot.dat file, i.e. columns
    3 and 4. Return the indices.
    """
    indices = load_column_file_to_nodes(filename, [2, 3])
    return indices


def load_mag_file(filename):
    index = load_column_file_to_elements(filename, [2, ])
    return index


def load_pha_file(filename):
    index = load_column_file_to_elements(filename, [2, ])
    return index


def load_elem_file(filename):
    """
    Determine the type (regular, irregular) of grid we are dealing with an call
    the corresponding load function.
    """
    if(not os.path.isfile(filename)):
        print('Grid file "{0}" not found'.format(filename))
        exit()
    if(nodes is not None):
        # grid already loaded
        return
    # TODO: do something here and implement irregular grids
    load_regular_elem_file(filename)


def load_regular_elem_file(filename):
    """
    Load an elem.dat file containing a regular grid, process it, and store
    the data in a global variable
    """
    global width_in_nodes
    global height_in_nodes
    f = open(filename)
    # first header line
    firstline = f.readline().lstrip()
    number_of_nodes, number_of_element_types, bandwidth = np.fromstring(
        firstline, dtype=int, sep='   ')

    # read header lines for each element type
    element_infos = np.zeros((number_of_element_types, 3), dtype=int)
    for element_type in range(0, number_of_element_types):
        element_line = f.readline().lstrip()
        element_infos[element_type] = np.fromstring(element_line, dtype=int,
                                                    sep='  ')

    # prepare nodes
    global nodes, nodes_sorted
    nodes_sorted = np.zeros((number_of_nodes, 3), dtype=float)
    nodes = np.zeros((number_of_nodes, 3), dtype=float)

    # read in nodes
    for node in range(0, number_of_nodes):
        node_line = f.readline().lstrip()
        nodes[node, :] = np.fromstring(node_line, dtype=float, sep='    ')

    # Rearrange nodes when CutMcK was used.
    # The check is based on the first node, but if one node was renumbered, so
    # were all the others.
    global nodes_cutmck_index
    nodes_cutmck_index = None
    if(nodes[0, 0] != 1 or nodes[1, 0] != 2):
        nodes_cutmck_index = np.zeros(nodes.shape[0])
        for node in range(0, number_of_nodes):
            new_index = np.where(nodes[:, 0].astype(int) == (node + 1))
            nodes_sorted[new_index[0], 1:3] = nodes[node, 1:3]
            nodes_sorted[new_index[0], 0] = new_index[0]
            nodes_cutmck_index[node] = new_index[0]
        print('This grid was sorted using CutMcK. The nodes were resorted!')
    else:
        nodes_sorted = nodes

    # round node coordinates to 5th decimal point. Sometimes this is important
    # when we deal with mal-formatted node data
    nodes_sorted[:, 1:3] = np.round(nodes_sorted[:, 1:3], 5)

    global element_type_list

    # read elements
    for element_type in range(0, number_of_element_types):
        element_list = []
        for element_coordinates in range(0,
                                         int(element_infos[element_type, 1])):
            element_coordinates_line = f.readline().lstrip()
            tmp_element = element()
            tmp_element.nodes = np.fromstring(element_coordinates_line,
                                              dtype=int, sep=' ')
            tmp_element.xcoords = nodes_sorted[tmp_element.nodes - 1, 1]
            tmp_element.zcoords = nodes_sorted[tmp_element.nodes - 1, 2]
            element_list.append(tmp_element)
        element_type_list.append(element_list)

    # depending on the type of grid (rectangular or triangle), prepare grids or
    # triangle lists
    global grid_is_rectangular
    if(len(element_type_list[0][0].nodes) == 3):
        print('Triangular grid found')
        grid_is_rectangular = False

        # prepare triangles
        global triangles
        triangles = element_type_list[0]
        triangles = [x.nodes for x in triangles]
        # python starts arrays with 0, but elem.dat with 1
        triangles = np.array(triangles) - 1

    else:
        print('Rectangular grid found')
        grid_is_rectangular = True
        # calculate the dimensions of the grid
        calculate_dimensions(nodes)

        # ok, we kind of ignore the actual elements and just create a regular
        # grid using the nodes.
        # TODO: This is a no-go if we ever encounter rectangular grids with
        #       topography
        xcoordinates = nodes[range(0, width_in_nodes), 1]
        ycoordinates = nodes[range(
            0,
            int(width_in_nodes * height_in_nodes),
            width_in_nodes),
            2
        ]

        meshx, meshy = np.meshgrid(xcoordinates, ycoordinates)

        # store in global variables
        global gridx
        global gridz
        gridx = meshx
        gridz = meshy


def calc_element_centroids():
    global elm_centroids
    elm_centroids = []

    # loop over element 8
    for element in element_type_list[0]:
        A = 0
        for i in range(0, 4):
            ind1 = i
            ind2 = (i + 1) % 4
            contrib = (element.xcoords[ind1] * element.zcoords[ind2] -
                       element.xcoords[ind2] * element.zcoords[ind1])
            A += contrib
        A *= 0.5

        x_centr = 0
        for i in range(0, 4):
            ind1 = i
            ind2 = (i + 1) % 4
            contrib = (element.xcoords[ind1] + element.xcoords[ind2]) *\
                      (element.xcoords[ind1] * element.zcoords[ind2] -
                       element.xcoords[ind2] * element.zcoords[ind1])
            x_centr += contrib
        x_centr *= 1 / (6 * A)

        z_centr = 0
        for i in range(0, 4):
            ind1 = i
            ind2 = (i + 1) % 4
            contrib = (element.zcoords[ind1] + element.zcoords[ind2]) * \
            (element.xcoords[ind1] * element.zcoords[ind2] - \
                    element.xcoords[ind2] * element.zcoords[ind1])
            z_centr += contrib
        z_centr *= 1 / (6 * A)

        elm_centroids.append((x_centr, z_centr))


def calculate_dimensions(nodes):
    """
    For a regular grid (rectangular cells), compute the dimensions in x and z
    directions (# nodes, # elements)
    """
    # define global variables
    global width_in_nodes, height_in_nodes
    global width_in_elements, height_in_elements

    # determine width and height of grid in element numbers
    width_in_nodes = 0
    initial_y_value = nodes[0, 2]
    for value in range(0, len(nodes[:, 2])):
        if(initial_y_value != nodes[value, 2]):
            break
        width_in_nodes += 1

    height_in_nodes = len(nodes) / width_in_nodes

    # store in global variables
    width_in_elements = width_in_nodes - 1
    height_in_elements = height_in_nodes - 1


def load_elec_file(filename):
    """
    Load an elec.dat file
    """
    if(not os.path.isfile(filename)):
        print('Grid electrode file "{0}" not found'.format(filename))
        exit()
    # TODO: Check if grid is already loaded
    global electrode_nodes
    electrode_nodes = np.genfromtxt(filename, dtype=int)

    global electrode_pos_x, electrode_pos_z
    electrode_pos_x = np.zeros((electrode_nodes[0]))
    electrode_pos_z = np.zeros((electrode_nodes[0]))

    # remove first line: number of electrodes
    electrode_nodes = np.delete(electrode_nodes, 0)

    for index, node in enumerate(electrode_nodes):
        electrode_pos_x[index] = nodes_sorted[node - 1, 1]
        electrode_pos_z[index] = nodes_sorted[node - 1, 2]


def get_dimensions_in_elements():
    """
    return the dimensions of the grid in element numbers, return (-1, -1) if no grid was loaded yet.
    """
    return width_in_elements, height_in_elements


def get_nr_of_elements():
    return (width_in_elements * height_in_elements)


def get_electrodes():
    """
    return electrode positions
    """
    return electrode_pos_x, electrode_pos_z


def get_grid():
    """
    Return the grid in a form usable by matplotlib functions.
    Treat the output as output from meshgrid!
    """
    return gridx, gridz


def get_nodes():
    """
    Return nodes
    """
    return nodes

