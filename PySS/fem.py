import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
from collections import namedtuple
import matplotlib.animation as animation

class FEModel:
    def __init__(self, name=None, hist_data=None):
        self.name = name
        self.hist_outs = hist_data

    def tuple2dict(self, data):
        """
        Used to convert the load-displacement data exported from models to a dictionary

        """
        ld_data = []
        for specimen in data:
            sp_dict = dict()
            load = []
            disp = []
            for action in specimen[0]:
                load.append(action[1])
            for action in specimen[1]:
                disp.append(action[1])

            sp_dict["Load"] = np.array(load)
            sp_dict["Disp"] = -1 * np.array(disp)
            ld_data.append(sp_dict)

    def plot_history(self, x_axis, y_axis):
        """
        XXXXXXXXXXXXXXXXXXXXXXXXXX
        """
        plt.figure()
        plt.plot(self.hist_outs[x_axis], self.hist_outs[y_axis])

    @classmethod
    def from_hist_pkl(cls, filename):
        """
        Creates an object and imports history output data.
        """
        with open(filename, "rb") as fh:
            history_data = pickle.load(fh)

        return cls(name=filename, hist_data=history_data)


class ParametricDB:
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

        Return
        ------
            ParametricDB

        """
        with open(filename, 'rU') as infile:
            reader = csv.reader(infile)
            db = {c[0]: c[1:] for c in zip(*reader)}

        for key in db.keys():
            if len(key.split("-")) > 1:
                n_dim = len(key.split("-"))
                dim_str = key
        dim_ticks = np.array([c.split(sep="-") for c in db[dim_str]])
        dim_lengths = [len(set(dim_ticks[:, i])) for i in range(n_dim)]
        dim_names = dim_str.split(sep="-")
        full_list = {i[0]: i[1:][0] for i in zip(dim_names, dim_ticks.T)}

        del db[dim_str]

        #df = pd.DataFrame(full_dict)

        Address = namedtuple("map", " ".join(dim_names))
        args = [tuple(sorted(set(dim_ticks[:, i]))) for i, j in enumerate(dim_names)]
        addressbook = Address(*args)

        mtx = {i: np.empty(dim_lengths) for i in db.keys()}
        #mtx = np.empty(dim_lengths)
        for responce in db.keys():
            for i, responce_value in enumerate(db[responce]):
                current_idx = tuple(addressbook[idx].index(full_list[name][i]) for idx ,name in enumerate(dim_names))
                mtx[responce][current_idx] = responce_value
            mtx[responce].flags.writeable = False

        return cls(addressbook, mtx)

    def get_slice(self, slice_at, response):
        """
        Get a slice of the database.

        Parameters
        ----------
        slice_at : dict of int
            A dictionary of the keys to be sliced at the assigned values.
        responce : str
            The name of the requested response to be sliced.

        """

        idx_arr = [0]*len(self.dimensions)

        for key in self.dimensions._fields:
            if key not in slice_at.keys():
                idx_arr[self.get_idx(key)] = slice(None, None)
        for name, value in zip(slice_at.keys(), slice_at.values()):
            idx_arr[self.get_idx(name)] = value

        return self.responses[response][idx_arr]

    def get_idx(self, attrname):
        """
        Get the index number of a parameter (dimension) in the database.

        Parameters
        ----------
        attrname : str

        """
        return(self.dimensions.index(self.dimensions.__getattribute__(attrname)))

    def contour_2d(self, slice_at, response, transpose=False, fig=None):
        """
        Contour plot.
        :param slice_at:
        :return:
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax  = fig.gca()

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

        sbplt = ax.contour(X.astype(np.float), Y.astype(np.float), Z.T)
        plt.clabel(sbplt, inline=1, fontsize=10)
        ttl = [i for i in zip(slice_at.keys(), ttl_values)]
        ttl = ", ".join(["=".join(i) for i in ttl])
        ax.set_title(response+" for : "+ttl)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return ax

    def animate_2d(self, slice_at, response, time=None):
        """
        Animate a contour plot for a parameter.

        Parameters
        ----------
        slice_at : dict
            Dictionary of keys corresponding slicing dimensions and theyr slicing values.

        Notes
        -----
        found @ https://stackoverflow.com/questions/23070305/how-can-i-make-an-animation-with-contourf/38401705
        by Iury Sousa

        """
        axes = [key for key in self.dimensions._fields if key not in slice_at.keys()]

        if time is None:
            time = axes[-1]

        data = self.get_slice(slice_at, response)
        length = data.shape[2]

        ax = self.contour_2d({**slice_at,**{time: 0}}, response)
        fig = plt.gcf()

        # X, Y = np.meshgrid(self.addressbook[self.get_idx(axes[0])], self.addressbook[self.get_idx(axes[1])])
        # Z = self.get_slice(slice_at, response)
        # ax.contour(X.astype(np.float), Y.astype(np.float), Z[:, :, 0].T)

        def animate(i):
            ax = plt.gca()
            ax.clear()
            ax = self.contour_2d({**slice_at, **{time: i}}, response, fig=fig)
            return ax

        interval = 1  # in seconds
        ani = animation.FuncAnimation(fig, animate, length, interval=interval*1e+3)

        plt.show()

        return ani
