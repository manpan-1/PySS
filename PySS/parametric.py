import sys
from itertools import product
from shutil import rmtree
# from shutil import copyfile
import os
import PySS.FileLock as flck
import csv
import numpy as np
import matplotlib
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
from labellines import labelLine, labelLines
# from cycler import cycler

matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'axes.titlesize': 11})
plt = matplotlib.pyplot
plt.rc('text', usetex=True)
#
# # Latex setup for plotting with titles and labels.
# params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
# plt.rcParams.update(params)
#
# params = {'text.latex.preamble': [r'\usepackage{xfrac}']}
# plt.rcParams.update(params)



class FactDB:
    """Read and manage databases holding results of factorial tables."""
    def __init__(self, dimensions, responses):
        self.responses = responses
        self.dimensions = dimensions

    @classmethod
    def from_file(cls, filename):
        """
        Create from file.

        The file should be comma separated, first row titles, subsequent rows only numbers.

        Parameters
        ----------
        filename : str
            Relative path/filename.

        Returns
        -------
        :obj:`FactorialDatabase`

        """
        # with open(filename, 'rU') as infile:
        #     reader = csv.reader(infile)
        #     n_dim = int(next(reader)[0].split()[0])
        #     db = {c[0]: c[1:] for c in zip(*reader)}

        with open(filename, 'rU') as infile:
            reader = csv.reader(infile, delimiter=";")
            n_dim = int(next(reader)[0].split()[0])
            db = [c for c in zip(*reader)]

        all_responses = {i[0]: i[1:] for i in db[n_dim:]}

        dim_ticks = np.array([i[1:] for i in db[:n_dim]]).T
        dim_lengths = [len(set(dim_ticks[:, i])) for i in range(n_dim)]
        dim_names = [db[i][0] for i in range(n_dim)]

        full_list = {i[0]: i[1:][0] for i in zip(dim_names, dim_ticks.T)}

        Address = namedtuple("map", " ".join(dim_names))
        args = [tuple(sorted(set(dim_ticks[:, i]))) for i, j in enumerate(dim_names)]
        dimensions = Address(*args)

        responses = {i: np.empty(dim_lengths) for i in all_responses.keys()}
        for response in all_responses.keys():
            for i, response_value in enumerate(all_responses[response]):
                current_idx = tuple(dimensions[idx].index(full_list[name][i]) for idx, name in enumerate(dim_names))
                responses[response][current_idx] = response_value
            responses[response].flags.writeable = False

        return cls(dimensions, responses)

    def to_file(self, filename):
        """
        Write the database to file.

        Parameters
        ----------
        filename : str

        """
        with open(filename, "w") as fh:
            ndimes = len(self.dimensions)
            fh.write(str(ndimes) + " dimensions\n")
            fh.write(";".join(self.dimensions._fields) + ";" + ";".join(self.responses.keys()) + "\n")

            # Create the combinations list
            combinations = tuple(product(*self.dimensions))

            # Write the values line by line
            for n, i in enumerate(combinations):
                fh.write(";".join(i) + ";" + ";".join([str(self.responses[rkey].flatten()[n]) for rkey in self.responses.keys()]) + "\n")

    def new_from_slice(self, slice_at, responses=None):
        """
        Create a new database from a slice of the current one.

        Parameters
        ----------
        slice_at : dict of int
            A dictionary of the keys to be sliced at the assigned values.
        responses : list of strings, optional
            The name of the requested response to be sliced. All responses are carried by default.

        Returns
        -------
        :obj:`FactorialDatabase`

        """
        if responses is None:
            responses = self.responses.keys()

        # Dimensions of the child database
        dim_names = [i for i in self.dimensions._fields if not i in slice_at.keys()]

        exclude = []
        for i in slice_at.keys():
            exclude.append(self.dimensions._fields.index(i))

        Address = namedtuple("map", " ".join(dim_names))
        args = [i for n, i in enumerate(self.dimensions) if n not in exclude]
        dimensions = Address(*args)

        # Build the slices of matrices for all the responses
        values = {response: self.get_slice(slice_at, response) for response in responses}

        return FactDB(dimensions, values)

    def get_slice(self, slice_at, response):
        """
        Get a slice of the database.

        Parameters
        ----------
        slice_at : dict of int
            A dictionary of the keys to be sliced at the assigned values.
        response : str
            The name of the requested response to be sliced.

        """

        idx_arr = [0] * len(self.dimensions)

        for key in self.dimensions._fields:
            if key not in slice_at.keys():
                idx_arr[self.get_idx(key)] = slice(None, None)
        for name, value in zip(slice_at.keys(), slice_at.values()):
            idx_arr[self.get_idx(name)] = value

        return self.responses[response][tuple(idx_arr)]

    def get_idx(self, attrname):
        """
        Get the index number of a parameter (dimension) in the database.

        Parameters
        ----------
        attrname : str

        """
        return (self.dimensions.index(self.dimensions.__getattribute__(attrname)))

    #TODO: docstring
    def plt_contours(self, slice_at, response, subplots_of=None, colorbar=True, transpose_sbplts=False, colormap=None):
        """
        Figure with multiple subplots.

        Parameters
        ----------
        :param slice_at:
        :param subplots_of:
        :param response:
        :return:

        """
        if subplots_of is None:
            subplots_of = "null"
            iter_dimension = [None]
            subplots = 1
            x_subplots = 1
            y_subplots = 1
        else:
            iter_dimension = self.dimensions[self.get_idx(subplots_of)]
            subplots = len(iter_dimension)
            n_divisors = tuple(divisors(subplots))
            x_subplots = n_divisors[len(n_divisors)//2]
            y_subplots = int(subplots/x_subplots)
            if transpose_sbplts:
                x_subplots, y_subplots = y_subplots, x_subplots

        if colormap is None:
            colormap = plt.cm.inferno

        sbplt = str(y_subplots) + str(x_subplots)

        # Start and configure a figure. The following are configurable values.
        x_padding = 0.6  # inches
        y_padding = 0.45  # inches

        pltax_x_width = 2.0  #inches
        pltax_y_width = 1.5  #inches

        relative_x_spacing = 0.02
        relative_y_spacing = 0.38

        x_spacing = relative_x_spacing * pltax_x_width  # inches
        y_spacing = relative_y_spacing * pltax_y_width  # inches

        if x_subplots == 1:
            x_inches_fig = 2 * x_padding + (1-relative_x_spacing/2)*pltax_x_width
        else:
            x_inches_fig = 2 * x_padding + pltax_x_width * x_subplots

        if y_subplots == 1:
            y_inches_fig = 2 * y_padding + (1-relative_y_spacing/2)*pltax_y_width
        else:
            y_inches_fig = 2 * y_padding + pltax_y_width * y_subplots

        relative_x_padding = x_padding / x_inches_fig
        relative_y_padding = y_padding / y_inches_fig

        fig = plt.figure(figsize=(x_inches_fig, y_inches_fig), dpi=100)

        fig.subplots_adjust(
            left=relative_x_padding,
            bottom=relative_y_padding,
            right=1-relative_x_padding,
            top=1-relative_y_padding,
            wspace=relative_x_spacing,
            hspace=relative_y_spacing
        )

        # print(
        #     relative_x_padding,
        #     relative_y_padding,
        #     1-relative_x_padding,
        #     1-relative_y_padding,
        #     relative_x_spacing,
        #     relative_y_spacing
        # )

        # Make a copy of the slice dimensions
        crrnt_slice = {}
        for i in slice_at:
            crrnt_slice[i] = slice_at[i]

        # Find max and min from all subplots
        vmax = self.get_slice(slice_at, response).max()
        vmin = self.get_slice(slice_at, response).min()

        #crrnt_slice[subplots_of] = None
        # Loop through subplots and plot
        for crrnt_sbplt, i in enumerate(iter_dimension):
            if i is not None:
                crrnt_slice[subplots_of] = crrnt_sbplt

            self.contour_2d(
                crrnt_slice,
                response,
                fig=fig,
                sbplt=int(sbplt + str(crrnt_sbplt + 1)),
                vmin=vmin,
                vmax=vmax,
                colormap=colormap
            )

        for n, i in enumerate(fig.axes):
            # i.set_xticks([10, 20, 30, 40])
            # i.set_yticks([30, 40, 50, 60])
            if not n/x_subplots == int(n/x_subplots):
                i.set_yticklabels([])
                i.set_ylabel("")
            if n < subplots - (subplots / y_subplots):
                i.set_xlabel("")

        if colorbar:
            m = plt.cm.ScalarMappable(cmap=colormap)
            m.set_array(np.linspace(vmin, vmax, num=5))
            cb_ax = fig.add_axes([
                1 - relative_x_padding + x_spacing/x_inches_fig,
                relative_y_padding,
                0.1 / x_inches_fig,
                1 - 2 * relative_y_padding
            ])  # [left, bottom, width, height]

            plt.colorbar(m, cax=cb_ax)

        return fig

    def contour_2d(
            self,
            slice_at,
            response,
            transpose=False,
            fig=None,
            sbplt=None,
            colorbar=False,
            title=None,
            colormap=None,
            **kwargs
    ):
        """
        Contour plot.

        Parameters
        ----------
        slice_at : dict
            Indices of the dimension values for which the array is sliced.
        response : str
            Response to be plotted.
        transpose : bool, optional
            Reverse the x-y plotted axes.
        fig : :obj:`matplotlib.figure`, optional
            Figure object to plot on. New figure created by default
        sbplt : int, optional
            Subplot description number. Default is 111, which implies a figure with a single plot on.

        """
        if fig is None:
            fig = plt.figure()

            if sbplt is None:
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(sbplt)
        else:
            if sbplt is None:
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(sbplt)

        if title is None:
            title = True

        axes = [key for key in self.dimensions._fields if key not in slice_at.keys()]

        if transpose:
            X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[1])], self.dimensions[self.get_idx(axes[0])])
            Z = self.get_slice(slice_at, response).T
            x_label, y_label = axes[1], axes[0]
        else:
            X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[0])], self.dimensions[self.get_idx(axes[1])])
            Z = self.get_slice(slice_at, response)
            x_label, y_label = axes[0], axes[1]

        if colormap is None:
            colormap = plt.cm.inferno

        ttl_values = [self.dimensions[self.get_idx(i)][slice_at[i]] for i in slice_at.keys()]

        # levels = np.arange(0, 2., 0.025)
        # sbplt = ax.contour(X.astype(np.float), Y.astype(np.float), Z.T, vmin=0.4, vmax=1., levels=levels, cmap=colormap)
        contour1 = ax.contour(X.astype(np.float), Y.astype(np.float), Z.T, cmap=plt.cm.gray_r, linewidths=1, **kwargs)
        contour2 = ax.contourf(X.astype(np.float), Y.astype(np.float), Z.T, cmap=colormap, **kwargs)
        plt.clabel(contour1, inline=1)
        if colorbar:
            cb_ax = fig.add_axes([0.86, 0.18, 0.02, 0.67])
            plt.colorbar(contour2, cax=cb_ax)

        if title:
            ttl = [i for i in zip(slice_at.keys(), ttl_values)]
            ttl = ", ".join(["=".join(i) for i in ttl])
            if ttl:
                ax.set_title("$" + response + "$" + " for : " + "$" + ttl + "$")
        ax.set_xlabel("$" + x_label + "$")
        ax.set_ylabel("$" + y_label + "$")
        ax.grid()

        return fig

    def diagram_plot_new(
            self,
            slice_at,
            response,
            transpose=False,
            ax=None,
            title=None,
            color=None,
            legend=None,
            labellines=None,
            **kwargs
    ):
        """
        Contour plot.

        Parameters
        ----------
        slice_at : dict
            Indices of the dimension values for which the array is sliced.
        response : str
            Response to be plotted.
        transpose : bool, optional
            Reverse the x-y plotted axes.
        fig : :obj:`matplotlib.figure`, optional
            Figure object to plot on. New figure created by default
        sbplt : int, optional
            Subplot description number. Default is 111, which implies a figure with a single plot on.

        """
        if ax is None:
            fig, ax = plt.subplots()

        if title is None:
            title = True

        axes = [key for key in self.dimensions._fields if key not in slice_at.keys()]

        if transpose:
            X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[1])], self.dimensions[self.get_idx(axes[0])])
            Z = self.get_slice(slice_at, response).T
            x_label, y_label = axes[1], axes[0]
        else:
            X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[0])], self.dimensions[self.get_idx(axes[1])])
            Z = self.get_slice(slice_at, response)
            x_label, y_label = axes[0], axes[1]

        if color is None:
            color = "black"

        if legend is None:
            legend = False

        if labellines is None:
            labellines = False

        ttl_values = [self.dimensions[self.get_idx(i)][slice_at[i]] for i in slice_at.keys()]

        # levels = np.arange(0, 2., 0.025)
        # sbplt = ax.contour(X.astype(np.float), Y.astype(np.float), Z.T, vmin=0.4, vmax=1., levels=levels, cmap=colormap)
        legends = []

        for n, j in enumerate(Y[:, 0]):
            ax.plot(np.asfarray(X[0, :], float), Z.T[n, :], color=color, label=j, **kwargs)
            legends.append("{} = {}".format(y_label, j))

        if labellines:
            labelLines(ax.get_lines(), zorder=2.5)

        plt.show()

        if legend:
            plt.legend(legends)

        if title:
            ttl = [i for i in zip(slice_at.keys(), ttl_values)]
            ttl = ", ".join(["=".join(i) for i in ttl])
            if ttl:
                ax.set_title("$" + response + "$" + " for : " + "$" + ttl + "$")
        ax.set_xlabel("$" + x_label + "$")
        ax.set_ylabel("$" + response + "$")
        ax.grid()

        return ax

    def diagram_plot(
            self,
            slice_at,
            response,
            transpose=False,
            fig=None,
            sbplt=None,
            title=None,
            color=None,
            legend=None,
            **kwargs
    ):
        """
        Contour plot.

        Parameters
        ----------
        slice_at : dict
            Indices of the dimension values for which the array is sliced.
        response : str
            Response to be plotted.
        transpose : bool, optional
            Reverse the x-y plotted axes.
        fig : :obj:`matplotlib.figure`, optional
            Figure object to plot on. New figure created by default
        sbplt : int, optional
            Subplot description number. Default is 111, which implies a figure with a single plot on.

        """


        if fig is None:
            fig = plt.figure()

            if sbplt is None:
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(sbplt)
        else:
            if sbplt is None:
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(sbplt)

        if title is None:
            title = True

        axes = [key for key in self.dimensions._fields if key not in slice_at.keys()]

        if transpose:
            X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[1])], self.dimensions[self.get_idx(axes[0])])
            Z = self.get_slice(slice_at, response).T
            x_label, y_label = axes[1], axes[0]
        else:
            X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[0])], self.dimensions[self.get_idx(axes[1])])
            Z = self.get_slice(slice_at, response)
            x_label, y_label = axes[0], axes[1]

        if color is None:
            color = "black"

        if legend is None:
            legend = True

        ttl_values = [self.dimensions[self.get_idx(i)][slice_at[i]] for i in slice_at.keys()]

        # levels = np.arange(0, 2., 0.025)
        # sbplt = ax.contour(X.astype(np.float), Y.astype(np.float), Z.T, vmin=0.4, vmax=1., levels=levels, cmap=colormap)
        legends = []
        for n, j in enumerate(Y[:, 0]):
            ax.plot(X[n, :], Z.T[n, :], color=color, **kwargs)
            legends.append("{} = {}".format(y_label, j))

        if legend:
            plt.legend(legends)

        if title:
            ttl = [i for i in zip(slice_at.keys(), ttl_values)]
            ttl = ", ".join(["=".join(i) for i in ttl])
            if ttl:
                ax.set_title("$" + response + "$" + " for : " + "$" + ttl + "$")
        ax.set_xlabel("$" + x_label + "$")
        ax.set_ylabel("$" + response + "$")
        ax.grid()

        return fig

    def surf_3d(self, slice_at, response, transpose=False, fig=None, sbplt=None, title=None, scale=None, colormap=None):
        """
        Surface plot.

        Parameters
        ----------
        slice_at : dict
            Indices of the dimension values for which the array is sliced.
        response : str
            Response to be plotted.
        transpose : bool, optional
            Reverse the x-y plotted axes.
        fig : :obj:`matplotlib.figure`, optional
            Figure object to plot on. New figure created by default
        sbplt : int, optional
            Subplot description number. Default is 111, which implies a figure with a single plot on.
        scale : (float, float, float), optional
            Scale factors for the three axes, `x y z`

        """
        # Convenient window dimensions
        # one subplot:
        # 2 side by side: Bbox(x0=0.0, y0=0.0, x1=6.79, y1=2.57)
        # azim elev =  -160  30
        # 3 subplots side by side
        # 4 subplots: Bbox(x0=0.0, y0=0.0, x1=6.43, y1=5.14)
        # 4 subplots: Bbox(x0=0.0, y0=0.0, x1=8, y1=6.5)
        # azim elev -160 30
        plt.rc('text', usetex=True)
        if fig is None:
            fig = plt.figure()
            if sbplt is None:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(sbplt, projection='3d')
        else:
            if sbplt is None:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(sbplt, projection='3d')

        if scale is None:
            scale = (1, 1, 1)

        if title is None:
            title = True

        if colormap is None:
            colormap = plt.cm.inferno

        axes = [key for key in self.dimensions._fields if key not in slice_at.keys()]

        if transpose:
            X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[1])], self.dimensions[self.get_idx(axes[0])])
            Z = self.get_slice(slice_at, response).T
            x_label, y_label = axes[1], axes[0]
        else:
            X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[0])], self.dimensions[self.get_idx(axes[1])])
            Z = self.get_slice(slice_at, response)
            x_label, y_label = axes[0], axes[1]

        ttl_values = [self.dimensions[self.get_idx(i)][slice_at[i]] for i in slice_at.keys()]

        sbplt = ax.plot_surface(
            X.astype(np.float) * scale[0],
            Y.astype(np.float) * scale[1],
            Z.T * scale[2],
            cmap=colormap
        )
        # plt.clabel(sbplt, inline=1, fontsize=10)
        ttl = [i for i in zip(slice_at.keys(), ttl_values)]
        ttl = ", ".join(["=".join(i) for i in ttl])
        if title:
            ax.set_title("$" + response + "$" + " for : " + "$" + ttl + "$")

        ax.set_xlabel("$" + x_label + "$")
        ax.set_ylabel("$" + y_label + "$")

        return fig

    @staticmethod
    def match_viewports(fig=None):
        """
        Matches the viewport of all the subplots in a given figure to the first subplot.

        Parameters
        ----------
        :param fig : :obj:`matplotlib.figure`, optional
            Figure with 3D subplots. Current figure used by default.

        """
        if fig is None:
            fig = plt.gcf()

        for i in fig.axes:
            i.view_init(azim=fig.axes[0].azim, elev=fig.axes[0].elev)


def dummyfunc(*args, **kargs):
    """ Null function used for testing :func:`~PySS.parametric.run_factorial`"""
    return("")


def divisors(n):
    """
    Divisors of an integer.

    Return all the possible divisors for a given integer.

    Parameters
    ----------
    n: int

    Returns
    -------
    int

    Notes
    -----

    """
    large_divisors = []
    for i in range(1, int(n**.5 + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(int(n / i))
    for divisor in reversed(large_divisors):
        yield divisor


def get_queued(filename):
    """
    Read a batch info file and return the indexes of the queued jobs.

    Parameters
    ----------
    filename : str
        Filename of the batch status file.

    """
    queue = []
    with flck.FileLock(filename):
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if sys.version[0] == "2":
                    curr_line = line.split(";")
                else:
                    curr_line = line.split(sep=";")
                if curr_line[-1] == "QUEUED\n":
                    queue.append(i - 2)
    return queue


def goto_next_queued(filename):
    """
    Read a batch info file and return the indexes of the queued jobs.

    Parameters
    ----------
    filename : str
        Filename of the batch status file.

    """
    with flck.FileLock(filename):
        lines = open(filename, "r").readlines()
        for i, line in enumerate(lines):
            if sys.version[0] == "2":
                curr_line = line.split(";")
            else:
                curr_line = line.split(sep=";")
            if curr_line[-1] == "QUEUED\n":
                lines[i] = ";".join(curr_line[:-1]) + ";RUNNING\n"
                with open(filename, "w") as f:
                    f.writelines(lines)
                return i - 2
        return False


def update_job_status(filename, job_nr, new_status):
    """
    Change status of a job number on a batch status file.

    Parameters
    ----------
    filename : str
        Filename of the batch status file.
    job_nr : int
        Index of the job to be updated.
    new_status : str
        String to be written as new status.

    """

    line_nr = job_nr + 2
    with flck.FileLock(filename):
        lines = open(filename, "r").readlines()
        if sys.version[0] == "2":
            curr_line = lines[line_nr].split(";")
        else:
            curr_line = lines[line_nr].split(sep=";")
        lines[line_nr] = ";".join(curr_line[:-1]) + ";" + new_status + "\n"
        with open(filename, "w") as f:
            f.writelines(lines)


def run_factorial(
    prj_name,
    exec_func,
    func_args=None,
    func_kargs=None,
    p_args=None,
    p_kargs=None,
    mk_subdirs=None,
    del_subdirs=None,
):
    """
    Run a function for a full factorial table of input parameters.

    Creates a full factorial matrix for a given list of parameters and executes a function for each combination. A list
    of values is needed for each parameter.

    Parameters
    ----------
    prj_name : str
        Name of the parametric project. Used for directory name and filenames.
    exec_func : function
        Function to be executed parametrically.
    func_args : list, optional
        A list containing the arguments to be passed to the function, both parametric and static.
    func_kargs : dict, optional
        A dictionaty of all the keyword arguments to be passed to the function, both parametric and static.
    p_args : list of integers, optional
        A list of integers, indicating the positions on the func_args list of the arguments for which the parametric
        matrix is composed. On the positions indicated by p_args, the func_args must contain list items.
    p_kargs : list of strings, optional
        A list of the keys of all the keyword arguments that need to be executed parametrically, similarly to p_args.
    mk_subdirs : bool, optional
        If `True`, each combination is executed in a different, automatically created, subdirectory.
        Useful when the executed job generates files on the working directory. If subdirectory with the same name
        exists, the job will enter and execute in the directory ignoring pre-existing files.
        Default is False (run all jobs in the current directory)
    del_subdirs : bool, optional
        Remove the subdirectories for the individual jobs and keep only the results summary.
        This argument is used in combination with the mk_subdirs.
        Default is False (do not remove).

    Returns
    -------
    list
        List item containing the results of all the executed runs

    """
    # Defaults
    if p_args is None:
        p_args = []
        
    if p_kargs is None:
        p_kargs = []

    if mk_subdirs is None:
        mk_subdirs = False

    if del_subdirs is None:
        del_subdirs = False

    # Pick up the parametric variables from the list of arguments
    if func_args:
        param_args = [func_args[x] for x in p_args]
    else:
        param_args = []

    if func_kargs:
        param_kargs = [func_kargs[x] for x in p_kargs]
    else:
        param_kargs = []

    print("The parametric argument values are: \n", param_args)
    print("The parametric keyword argument values are: \n", param_kargs)

    # Cartesian product of parameters
    all_params = param_args + param_kargs
    combinations = list(product(*all_params))

    # Check which of the given parametric arguments are numeric and find our how many leading zeros are needed based on
    # the largest value.
    isnumeric = lambda x: isinstance(x, int) or isinstance(x, float)
    numeric_dimensions = [param_args.index(x) for x in param_args if isnumeric(max(x))]
    leading_zeros = [len(str(max(x))) for x in param_args if isnumeric(max(x))]
    
    # Write the combinations in a file
    batch_status_file = prj_name + "_batch_status.csv"
    if not os.path.isfile(batch_status_file):
        with open(batch_status_file, "w") as f:
            f.write(str(len(param_args)+len(param_kargs)) + " dimensions\n")
            f.write("job_parameters;status\n")
            for dimensions in combinations:
                dim_string = ""
                for idx, dimension in enumerate(dimensions):
                    if idx in numeric_dimensions:
                        dim_string = dim_string + str(dimension).zfill(leading_zeros[idx]) + ";"
                    else:
                        dim_string = dim_string + str(dimension) + ";"
                dim_string = dim_string + "QUEUED" + "\n"
                f.write(dim_string)
    
    # Initiate (if doesn't exist) a file for the results
    results_file = prj_name + "_results.csv"
    if not os.path.isfile(results_file):
        with open(results_file, "w") as f:
            f.write(str(len(param_args)+len(param_kargs)) + " dimensions\n")
            f.write("(For this database to be loaded with the PySS.fem.ParametricDB(), replace this line with the "
                    "titles of the columns in a semicolon separated list. TeX expressions can be used.)\n")
            for dimensions in combinations:
                dim_string = ""
                for idx, dimension in enumerate(dimensions):
                    if idx in numeric_dimensions:
                        dim_string = dim_string + str(dimension).zfill(leading_zeros[idx]) + ";"
                    else:
                        dim_string = dim_string + str(dimension) + ";"
                dim_string = dim_string + "Wait for it...!" + "\n"
                f.write(dim_string)
    # Initiate a list for the results
    prj_results = []

    while True:
        curr_job_nr = goto_next_queued(batch_status_file)
        if curr_job_nr or curr_job_nr is 0:
            comb_case = combinations[curr_job_nr]
            # Construct an id string for the current job based on the combination. The string is formatted so that it
            # does contain any illegal characters for filename and abaqus job name usage.
            if sys.version[0] == "2":
                job_id = str(comb_case).translate(None, ",.&*~!()[]{}|;:\'\"`<>?/\\")
            else:
                job_id = str(comb_case).translate(str.maketrans("", "", ",.&*~!()[]{}|;:\'\"`<>?/\\"))
            job_id = job_id.replace(" ", "-")
            job_id = job_id + "-" + prj_name
    
            # # Check if the directory exists. If so, continue to the next job.
            # if os.path.isdir("./" + job_id):
            #     print("Job already exists: A directory with the same name'" + job_id + "' exists in the cwd")
            #     continue
    
            if mk_subdirs:
                if os.path.isdir(job_id):
                    os.chdir(job_id)
                else:
                    # Make a new subdirectory for the current session
                    os.mkdir(job_id)
    
                    # Change working directory
                    os.chdir(job_id)
                 
                # copyfile("../abq_env_" + os.environ["SLURM_JOBID"] + ".env", "./abaqus_v6.env")
            
            # Assemble current job's input arguments
            if func_args:
                current_func_args = func_args
                for (index, new_parameter) in zip(p_args, comb_case):
                    current_func_args[index] = new_parameter
            else:
                current_func_args = []
    
            # Assemble current job's input keyword arguments
            if func_kargs:
                current_func_kargs = func_kargs
                for i, x in enumerate(p_kargs):
                    current_func_kargs[x] = comb_case[len(p_args) + i]
            else:
                current_func_kargs = {}

            current_func_kargs["IDstring"] = job_id
    
            # The function to be run for the full factorial parametric is called here
            print("Running job nr: " + str(curr_job_nr) + " with name: " + job_id)
    
            # Execute the current job
            try:
                result = exec_func(*current_func_args, **current_func_kargs)
                new_status = "COMPLETED"
            except:
                result = []
                new_status = "FAILED"
    
            # Create an output string
            result = str(result)
    
            # Return to parent directory
            if mk_subdirs is True:
                os.chdir('../')

            # Add the result of the current job to the return list
            prj_results = prj_results + [result]
    
            # Remove job's folder (only the output information is kept)
            if mk_subdirs and del_subdirs is True:
                rmtree(job_id)

            # Write the result of the current job to the file
            update_job_status(results_file, curr_job_nr, result)
            
            # Update the job status on the common batch status file
            update_job_status(batch_status_file, curr_job_nr, new_status)
        else:
            break
        
    return prj_results


def subdir_crawler(
        exec_func,
        func_args=None,
        func_kargs=None,
        prj_name=None,):
    """
    Execute function in all subdirs.

    Parameters
    ----------
    exec_func : function
        Function to be executed parametrically.
    func_args : list, optional
        List of arguments for the function. No arguments by default.
    func_kargs : dict, optional
        Dictionary of keyword arguments for the function. No keyword arguments by default.
    prj_name : str, optional
        Project name.

    """
    if func_args is None:
        func_args = ()

    if func_kargs is None:
        func_kargs = {}

    if prj_name is None:
        prj_name = ""

    all_dirs = [i for i in os.listdir('.') if os.path.isdir(i)]
    all_dirs.sort()

    for directory in all_dirs:
        os.chdir(directory)
        # Execute the current job
        try:
            func_return = exec_func(*func_args, **func_kargs)
        except:
            func_return = "FAILED"

        os.chdir("..")
        with open('./' + prj_name + '_info.csv', 'a') as out_file:
            out_file.write(directory + ";" + str(func_return) + "\n")
