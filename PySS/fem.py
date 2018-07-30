import matplotlib.pyplot as plt
import numpy as np
import pickle
# import csv
# from collections import namedtuple
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation as animation
# import matplotlib.colors as mc


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

#
# class ParametricDB:
#     def __init__(self, dimensions, responses):
#         self.responses = responses
#         self.dimensions = dimensions
#
#     @classmethod
#     def from_file(cls, filename):
#         """
#         Create from file.
#
#         The file should be comma separated, first row titles, subsequent rows only numbers.
#
#         Parameters
#         ----------
#         filename : str
#             Relative path/filename.
#
#         Return
#         ------
#             ParametricDB
#
#         """
#         # with open(filename, 'rU') as infile:
#         #     reader = csv.reader(infile)
#         #     n_dim = int(next(reader)[0].split()[0])
#         #     db = {c[0]: c[1:] for c in zip(*reader)}
#
#         with open(filename, 'rU') as infile:
#             reader = csv.reader(infile, delimiter=";")
#             n_dim = int(next(reader)[0].split()[0])
#             db = [c for c in zip(*reader)]
#
#         all_responses = {i[0]: i[1:] for i in db[n_dim:]}
#
#         dim_ticks = np.array([i[1:] for i in db[:n_dim]]).T
#         dim_lengths = [len(set(dim_ticks[:, i])) for i in range(n_dim)]
#         dim_names = [db[i][0] for i in range(n_dim)]
#
#         # with open(filename, 'r') as infile:
#         #     all_lines = [[c.split(sep=":")[0]] + c.split(sep=":")[1].split(sep=",") for c in infile]
#         #     db = {c[0]: c[1:] for c in zip(*all_lines)}
#
#         # for key in db.keys():
#         #     if len(key.split(",")) > 1:
#         #         n_dim = len(key.split(","))
#         #         dim_str = key
#         # dim_ticks = np.array([c.split(sep=",") for c in db[dim_str]])
#         # dim_lengths = [len(set(dim_ticks[:, i])) for i in range(n_dim)]
#         # dim_names = dim_str.split(sep=",")
#         full_list = {i[0]: i[1:][0] for i in zip(dim_names, dim_ticks.T)}
#
#         # del db[dim_str]
#
#         #df = pd.DataFrame(full_dict)
#
#         Address = namedtuple("map", " ".join(dim_names))
#         args = [tuple(sorted(set(dim_ticks[:, i]))) for i, j in enumerate(dim_names)]
#         addressbook = Address(*args)
#
#         mtx = {i: np.empty(dim_lengths) for i in all_responses.keys()}
#         for response in all_responses.keys():
#             for i, response_value in enumerate(all_responses[response]):
#                 current_idx = tuple(addressbook[idx].index(full_list[name][i]) for idx, name in enumerate(dim_names))
#                 mtx[response][current_idx] = response_value
#             mtx[response].flags.writeable = False
#
#         return cls(addressbook, mtx)
#
#     def get_slice(self, slice_at, response):
#         """
#         Get a slice of the database.
#
#         Parameters
#         ----------
#         slice_at : dict of int
#             A dictionary of the keys to be sliced at the assigned values.
#         response : str
#             The name of the requested response to be sliced.
#
#         """
#
#         idx_arr = [0]*len(self.dimensions)
#
#         for key in self.dimensions._fields:
#             if key not in slice_at.keys():
#                 idx_arr[self.get_idx(key)] = slice(None, None)
#         for name, value in zip(slice_at.keys(), slice_at.values()):
#             idx_arr[self.get_idx(name)] = value
#
#         return self.responses[response][idx_arr]
#
#     def get_idx(self, attrname):
#         """
#         Get the index number of a parameter (dimension) in the database.
#
#         Parameters
#         ----------
#         attrname : str
#
#         """
#         return(self.dimensions.index(self.dimensions.__getattribute__(attrname)))
#
#     def contour_2d(self, slice_at, response, transpose=False, fig=None, sbplt=None):
#         """
#         Contour plot.
#         :param slice_at:
#         :return:
#         """
#         plt.rc('text', usetex=True)
#         if fig is None:
#             fig = plt.figure()
#             if sbplt is None:
#                 ax = fig.add_subplot(111)
#             else:
#                 ax = fig.add_subplot(sbplt)
#         else:
#             if sbplt is None:
#                 ax = fig.add_subplot(111)
#             else:
#                 ax = fig.add_subplot(sbplt)
#
#         axes = [key for key in self.dimensions._fields if key not in slice_at.keys()]
#
#         if transpose:
#             X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[1])], self.dimensions[self.get_idx(axes[0])])
#             Z = self.get_slice(slice_at, response).T
#             x_label, y_label = axes[1], axes[0]
#         else:
#             X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[0])], self.dimensions[self.get_idx(axes[1])])
#             Z = self.get_slice(slice_at, response)
#             x_label, y_label = axes[0], axes[1]
#
#         ttl_values = [self.dimensions[self.get_idx(i)][slice_at[i]] for i in slice_at.keys()]
#
#         # levels = np.arange(0, 2., 0.025)
#         # sbplt = ax.contour(X.astype(np.float), Y.astype(np.float), Z.T, vmin=0.4, vmax=1., levels=levels, cmap=plt.cm.inferno)
#         sbplt = ax.contour(X.astype(np.float), Y.astype(np.float), Z.T, cmap=plt.cm.gray_r)
#         sbplt2 = ax.contourf(X.astype(np.float), Y.astype(np.float), Z.T, cmap=plt.cm.inferno)
#         plt.clabel(sbplt, inline=1, fontsize=10)
#         ttl = [i for i in zip(slice_at.keys(), ttl_values)]
#         ttl = ", ".join(["=".join(i) for i in ttl])
#         ax.set_title("$" + response + "$" + " for : " + "$" + ttl + "$")
#         ax.set_xlabel("$"+x_label+"$")
#         ax.set_ylabel("$"+y_label+"$")
#
#         return fig
#
#     def surf_3d(self, slice_at, response, transpose=False, fig=None, sbplt=None):
#         """
#         Surface plot.
#         :param slice_at:
#         :return:
#         """
#         #Convenient window dimensions
#         # one subplot:
#         # 2 side by side: Bbox(x0=0.0, y0=0.0, x1=6.79, y1=2.57)
#         # azim elev =  -160  30
#         # 3 subplots side by side
#         # 4 subplots: Bbox(x0=0.0, y0=0.0, x1=6.43, y1=5.14)
#         #azim elev -160 30
#         plt.rc('text', usetex=True)
#         if fig is None:
#             fig = plt.figure()
#             if sbplt is None:
#                 ax = fig.add_subplot(111, projection='3d')
#             else:
#                 ax = fig.add_subplot(sbplt, projection='3d')
#         else:
#             if sbplt is None:
#                 ax = fig.add_subplot(111, projection='3d')
#             else:
#                 ax = fig.add_subplot(sbplt, projection='3d')
#
#
#         axes = [key for key in self.dimensions._fields if key not in slice_at.keys()]
#
#         if transpose:
#             X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[1])], self.dimensions[self.get_idx(axes[0])])
#             Z = self.get_slice(slice_at, response).T
#             x_label, y_label = axes[1], axes[0]
#         else:
#             X, Y = np.meshgrid(self.dimensions[self.get_idx(axes[0])], self.dimensions[self.get_idx(axes[1])])
#             Z = self.get_slice(slice_at, response)
#             x_label, y_label = axes[0], axes[1]
#
#         ttl_values = [self.dimensions[self.get_idx(i)][slice_at[i]] for i in slice_at.keys()]
#
#         sbplt = ax.plot_surface(X.astype(np.float), Y.astype(np.float), Z.T, cmap=plt.cm.inferno)
#         # plt.clabel(sbplt, inline=1, fontsize=10)
#         ttl = [i for i in zip(slice_at.keys(), ttl_values)]
#         ttl = ", ".join(["=".join(i) for i in ttl])
#         ax.set_title("$" + response + "$" + " for : " + "$" + ttl + "$")
#         ax.set_xlabel("$"+x_label+"$")
#         ax.set_ylabel("$"+y_label+"$")
#
#         return fig
#
# def match_viewports(fig=None):
#     if fig is None:
#         fig = plt.gcf()
#     fig.axes[1].view_init(azim=fig.axes[0].azim, elev=fig.axes[0].elev)


def main():
    lambda01 = ParametricDB.from_file("data/fem/fem-results_lambda01.dat")
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcA, f_yield: 355 MPa, lambda_flex: 0.1")
    lambda01.contour_2d({"plate_imp": 0, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[0, 0])
    lambda01.contour_2d({"plate_imp": 1, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[0, 1])
    lambda01.contour_2d({"plate_imp": 2, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[0, 2])
    lambda01.contour_2d({"plate_imp": 3, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[1, 0])
    lambda01.contour_2d({"plate_imp": 4, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[1, 1])
    lambda01.contour_2d({"plate_imp": 5, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[1, 2])
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcB, f_yield: 355 MPa, lambda_flex: 0.1")
    lambda01.contour_2d({"plate_imp": 0, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[0, 0])
    lambda01.contour_2d({"plate_imp": 1, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[0, 1])
    lambda01.contour_2d({"plate_imp": 2, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[0, 2])
    lambda01.contour_2d({"plate_imp": 3, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[1, 0])
    lambda01.contour_2d({"plate_imp": 4, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[1, 1])
    lambda01.contour_2d({"plate_imp": 5, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[1, 2])
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcC, f_yield: 355 MPa, lambda_flex: 0.1")
    lambda01.contour_2d({"plate_imp": 0, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[0, 0])
    lambda01.contour_2d({"plate_imp": 1, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[0, 1])
    lambda01.contour_2d({"plate_imp": 2, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[0, 2])
    lambda01.contour_2d({"plate_imp": 3, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[1, 0])
    lambda01.contour_2d({"plate_imp": 4, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[1, 1])
    lambda01.contour_2d({"plate_imp": 5, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[1, 2])

    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcA, f_yield: 700 MPa, lambda_flex: 0.1")
    lambda01.contour_2d({"plate_imp": 0, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[0, 0])
    lambda01.contour_2d({"plate_imp": 1, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[0, 1])
    lambda01.contour_2d({"plate_imp": 2, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[0, 2])
    lambda01.contour_2d({"plate_imp": 3, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[1, 0])
    lambda01.contour_2d({"plate_imp": 4, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[1, 1])
    lambda01.contour_2d({"plate_imp": 5, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[1, 2])
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcB, f_yield: 700 MPa, lambda_flex: 0.1")
    lambda01.contour_2d({"plate_imp": 0, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[0, 0])
    lambda01.contour_2d({"plate_imp": 1, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[0, 1])
    lambda01.contour_2d({"plate_imp": 2, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[0, 2])
    lambda01.contour_2d({"plate_imp": 3, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[1, 0])
    lambda01.contour_2d({"plate_imp": 4, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[1, 1])
    lambda01.contour_2d({"plate_imp": 5, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[1, 2])
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcC, f_yield: 700 MPa, lambda_flex: 0.1")
    lambda01.contour_2d({"plate_imp": 0, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[0, 0])
    lambda01.contour_2d({"plate_imp": 1, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[0, 1])
    lambda01.contour_2d({"plate_imp": 2, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[0, 2])
    lambda01.contour_2d({"plate_imp": 3, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[1, 0])
    lambda01.contour_2d({"plate_imp": 4, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[1, 1])
    lambda01.contour_2d({"plate_imp": 5, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[1, 2])




    lambda02 = ParametricDB.from_file("data/fem/fem-results-lambda02.dat")
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcA, f_yield: 355 MPa, lambda_flex: 0.2")
    lambda02.contour_2d({"plate_imp": 0, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[0, 0])
    lambda02.contour_2d({"plate_imp": 1, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[0, 1])
    lambda02.contour_2d({"plate_imp": 2, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[0, 2])
    lambda02.contour_2d({"plate_imp": 3, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[1, 0])
    lambda02.contour_2d({"plate_imp": 4, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[1, 1])
    lambda02.contour_2d({"plate_imp": 5, "fab_class": 0, "f_yield": 0}, "lpf", ax=ax[1, 2])
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcB, f_yield: 355 MPa, lambda_flex: 0.2")
    lambda02.contour_2d({"plate_imp": 0, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[0, 0])
    lambda02.contour_2d({"plate_imp": 1, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[0, 1])
    lambda02.contour_2d({"plate_imp": 2, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[0, 2])
    lambda02.contour_2d({"plate_imp": 3, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[1, 0])
    lambda02.contour_2d({"plate_imp": 4, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[1, 1])
    lambda02.contour_2d({"plate_imp": 5, "fab_class": 1, "f_yield": 0}, "lpf", ax=ax[1, 2])
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcC, f_yield: 355 MPa, lambda_flex: 0.2")
    lambda02.contour_2d({"plate_imp": 0, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[0, 0])
    lambda02.contour_2d({"plate_imp": 1, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[0, 1])
    lambda02.contour_2d({"plate_imp": 2, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[0, 2])
    lambda02.contour_2d({"plate_imp": 3, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[1, 0])
    lambda02.contour_2d({"plate_imp": 4, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[1, 1])
    lambda02.contour_2d({"plate_imp": 5, "fab_class": 2, "f_yield": 0}, "lpf", ax=ax[1, 2])

    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcA, f_yield: 700 MPa, lambda_flex: 0.2")
    lambda02.contour_2d({"plate_imp": 0, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[0, 0])
    lambda02.contour_2d({"plate_imp": 1, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[0, 1])
    lambda02.contour_2d({"plate_imp": 2, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[0, 2])
    lambda02.contour_2d({"plate_imp": 3, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[1, 0])
    lambda02.contour_2d({"plate_imp": 4, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[1, 1])
    lambda02.contour_2d({"plate_imp": 5, "fab_class": 0, "f_yield": 1}, "lpf", ax=ax[1, 2])
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcB, f_yield: 700 MPa, lambda_flex: 0.2")
    lambda02.contour_2d({"plate_imp": 0, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[0, 0])
    lambda02.contour_2d({"plate_imp": 1, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[0, 1])
    lambda02.contour_2d({"plate_imp": 2, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[0, 2])
    lambda02.contour_2d({"plate_imp": 3, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[1, 0])
    lambda02.contour_2d({"plate_imp": 4, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[1, 1])
    lambda02.contour_2d({"plate_imp": 5, "fab_class": 1, "f_yield": 1}, "lpf", ax=ax[1, 2])
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("fab_class: fcC, f_yield: 700 MPa, lambda_flex: 0.2")
    lambda02.contour_2d({"plate_imp": 0, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[0, 0])
    lambda02.contour_2d({"plate_imp": 1, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[0, 1])
    lambda02.contour_2d({"plate_imp": 2, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[0, 2])
    lambda02.contour_2d({"plate_imp": 3, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[1, 0])
    lambda02.contour_2d({"plate_imp": 4, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[1, 1])
    lambda02.contour_2d({"plate_imp": 5, "fab_class": 2, "f_yield": 1}, "lpf", ax=ax[1, 2])

    return
