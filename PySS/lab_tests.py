# -*- coding: utf-8 -*-

"""
Module with functions related to laboratory work and data acquired with CATMAN.

"""

import numpy as np
import csv
import codecs
from os.path import basename, splitext
from matplotlib import pyplot as plt
import PySS.analytic_geometry as ag


class Experiment:
    """
    Laboratory test data.

    Class laboratory experiment containing methods for loading and manipulating data recorded with CATMAN software. The
    class is designed to be used with the `from_...` alternative constructors and not by giving the data directly for
    the object to be constructed.

    """
    def __init__(self, name=None, channels=None):
        if channels is None:
            self.channels = {}
        elif not isinstance(channels, dict):
            print("Channels must be a dictionary.")
            return NotImplemented
        else:
            self.channels = channels
        self.name = name

    def plot2d(
            self,
            x_data,
            y_data,
            ax=None,
            scale=None,
            axlabels=True,
            reduced=1,
            **kargs
    ):
        """
        Plot two recorded channels against each other.

        Parameters
        ----------
        x_data : str
            Key from self.data dictionary for the x-axis.
        y_data : str
            Key from self.data dictionary for the y-axis.
        ax : axis object, optional
            The axis to be used for plotting. By default, new figure and axis are created.
        scale : list or tuple of 2 numbers
            Scaling factors for `x` and `y` axis
        axlabels : bool, optional
            Write labels for x and y axes based on the channel name. TeX is used so certain names may cause problems in
            which case the used might want to deactivate plotting axes labels. Default is True.
        reduced : float, optional
            Plot a subset of points, the size of which is given as a fraction of the full set. Default value is 1.0
            which implies all points are plotted.

        """
        #plt.rc('text', usetex=False)

        if ax is None:
            fig, ax = plt.subplots()
        
        if scale is None:
            scale = (1, 1)

        ax.plot(
            self.channels[x_data]["data"][0::int(1/reduced)]*scale[0],
            self.channels[y_data]["data"][0::int(1/reduced)]*scale[1],
            label=self.name,
            **kargs
        )

        if axlabels:
            plt.xlabel(x_data + ", [" + self.channels[x_data]["units"] + "]")
            plt.ylabel(y_data + ", [" + self.channels[y_data]["units"] + "]")
            plt.title(self.name)

        return ax

    def add_new_channel_zeros(self, ch_name, units):
        """"
        Initialise a new channel entry in the dictionary with zeros.

        Parameters
        ----------
        ch_name : str
        units : str

        """
        self.channels[ch_name] = {}
        self.channels[ch_name]["data"] = np.zeros([len(self.channels[next(iter(self.channels))]["data"])])
        self.channels[ch_name]["units"] = units

    def invert_sign(self, ch_name):
        """Invert the sign on the values of a channel"""
        self.channels[ch_name]["data"] = -1 * self.channels[ch_name]["data"]

    @classmethod
    def from_file(cls, fh):
        """
        Alternative constructor.

        Imports data from a CATMAN output file. Uses 'ISO-8859-1' text encoding.

        Parameters
        ----------
        fh : str
            Path of ascii data file.

        """

        # Name from filename.
        filename = splitext(basename(fh))[0]

        # Open the requested file.
        f = codecs.open(fh, 'r', 'ISO-8859-1')

        # Read the header
        header = list(csv.reader([next(f) for x in range(7)], delimiter='\t'))

        # Read the column headers.
        next(f)
        column_head = list(csv.reader([next(f) for x in range(29)], delimiter='\t'))

        # Read the tab separated values.
        next(f)
        values = list(csv.reader(f, delimiter='\t'))

        # Get the number of channels
        n_chan = len(values[0])

        # Create a dictionary.
        channels = {}

        # Build a dictionary with the data using the column header to fetch the dict keys.
        for i in range(n_chan):
            data = np.empty(len(values))
            name = column_head[0][i].partition(' ')[0]
            channels[name] = {}
            units = column_head[1][i]
            channels[name]["units"] = units
            for j, row in enumerate(values):
                data[j] = float(row[i].replace(',', '.'))

            channels[name]["data"] = data

        return cls(name=filename, channels=channels)


class CouponTest(Experiment):
    """
    Class for coupon tests.

    Inherits the basic properties of a generic experiment class and offers standard calculations for rectangular
    cross-section tensile coupon tests.

    """

    def __init__(self, name=None, channels=None, thickness=None, width=None, l_0=None):
        super().__init__(name=name, channels=channels)
        self.thickness = thickness
        self.width = width
        self.l_0 = l_0
        self.initial_stiffness = None
        self.young = None
        # If geometric data exist, calculate the initial area.
        if (
            isinstance(thickness, float) or isinstance(thickness, int)
        ) and (
            isinstance(width, float) or isinstance(width, int)
        ):
            self.A_0 = thickness * width
        else:
            self.A_0 = None

    def clean_initial(self):
        """
        Remove head and tail of the load-displacement data.

        """

        # Filter the indices of the values larger then 1 kN, as a way of excluding head and tail recorded data.
        valid = np.where(self.channels["Load"]["data"] > 1)[0]

        for i in self.channels.values():
            i["data"] = i["data"][valid]

    def calc_init_stiffness(self):
        """
        Calculate the modulus of elasticity.

        The calculation is based on the range between 20% and 30% of the max load.

        """
        # Find the 20% and 30% of the max load
        lims = np.round([np.max(self.channels["Load"]["data"]) * 0.2, np.max(self.channels["Load"]["data"]) * 0.3]).astype(int)

        indices = [np.argmax(self.channels["Load"]["data"] > lims[0]), np.argmax(self.channels["Load"]["data"] > lims[1])]
        disp_el = self.channels["Epsilon"]["data"][indices[0]:indices[1]]
        load_el = self.channels["Load"]["data"][indices[0]:indices[1]]

        # fitting
        A = np.vstack([disp_el, np.ones(len(disp_el))]).T
        m, c = np.linalg.lstsq(A, load_el)[0]

        # Save the initial stiffness in th eobject
        self.initial_stiffness = m

        # Return the tangent
        return [m, c]

    def offset_to_0(self):
        """
        Offset the stroke values to start from 0 based on a regression on the initial stiffness.
        """

        # Linear regression on the elastic part to get stiffness and intercept
        m, c = self.calc_init_stiffness()
        # calculate the stroke offset
        offset = -c/m

        # Offset original values
        self.channels["Epsilon"]["data"] = self.channels["Epsilon"]["data"] - offset

        # Add the initial 0 values in both stroke and load arrays
        self.channels["Epsilon"]["data"] = np.concatenate([[0], self.channels["Epsilon"]["data"]])
        self.channels["Load"]["data"] = np.concatenate([[0], self.channels["Load"]["data"]])

    def add_initial_data(self, thickness, width, l_0):
        """
        Add initial geometrical data of the coupon specimen.

        Assuming that the imported test data provide stroke-load readings from testing, the geometric data of the coupon
        (thickness, width, initial gauge length) are needed to calculate the stress-strain curves. The method will
        overwrite pre existing.

        Parameters
        ----------
        thickness : float
            Coupon's measured thickness.
        width : float
            Coupon's measured width.
        l_0 : float
            Initial gauge length, %l_{0}%

        """
        self.thickness = thickness
        self.width = width
        self.l_0 = l_0
        self.A_0 = thickness * width

    def calc_stress_strain(self):
        """
        Calculate stress-strain curve.

        Calculate the engineering ant the true stress-strain curves for the coupon test. Initial geometric data as well
        as test data (stroke, load) must be assigned to the object for the method to run. The method assumes that
        entries with keys 'Stroke' and 'Load' exit in the data dictionary.

        """

        self.channels['StressEng'] = {"data": 1000 * self.channels['Load']["data"] / self.A_0,
                                      "units": "MPa"}
        self.channels['StrainEng'] = {"data": self.channels['Epsilon']["data"] / self.l_0,
                                      "units": "mm/mm"}

    def calc_young(self):
        """Calculate the modulus of elasticity."""
        if self.l_0 and self.A_0:
            if not self.initial_stiffness:
                self.calc_init_stiffness()

            self.young = 1000 * self.initial_stiffness * (self.l_0 / self.A_0)
        else:
            print("Missing information. Check if l_0 and A_0 exist.")

    def calc_plastic_strain(self):
        """
        Calculate the plastic strain.

        """
        self.channels["StrainPl"] = {"data": self.channels["StrainEng"]["data"] - self.channels["StressEng"]["data"] / self.young,
                                     "units": "mm/mm"}

    def calc_fy(self):
        """
        Calculate the yield stress based on the 0.2% plastic strain criterion.

        """
        line = ag.Line2D.from_2_points([0.2, 0], [0.2 + 900 / 2100, 900])
        for i, strain in enumerate(self.channels["StrainEng"]["data"] * 100):
            if (self.channels["StressEng"]["data"][i] - line.y_for_x(strain)) < 0:
                self.f_yield = self.channels["StressEng"]["data"][i]
                return

    def plot_stressstrain_eng(self, ax=None, **kargs):
        ax = self.plot2d('StrainEng', 'StressEng', scale=[100, 1], ax=ax, **kargs)
        plt.xlabel('Strain, $\\varepsilon_{eng}$ [\%]')
        plt.ylabel('Stress, $\\sigma_{eng}$ [MPa]')
        plt.title(self.name)
        return ax


def load_coupons():
    cp = {}

    # 2 mm plate
    widths = [20.408, 20.386, 20.397, 20.366, 20.35, 20.39]
    thicknesses = [1.884, 1.891, 1.9, 1.88, 1.882, 1.878]
    l_0 = 80
    for i in range(1, 7):
        coupon = CouponTest.from_file('./data/coupons/S700_2mm/cp{}.asc'.format(i))
        coupon.clean_initial()
        coupon.calc_init_stiffness()
        coupon.offset_to_0()
        coupon.add_initial_data(thicknesses[i-1], widths[i-1], l_0=l_0)
        coupon.calc_stress_strain()
        coupon.calc_young()
        coupon.calc_plastic_strain()
        coupon.calc_fy()

        cp["cp{}_2mm".format(i)] = coupon

    # 3 mm plate
    widths = [19.68, 19.63, 19.68, 19.716, 19.681, 19.76]
    thicknesses = [3.037, 3.031, 3.038, 3.051, 3.038, 3.036]


    for i in range(1, 7):
        coupon = CouponTest.from_file('./data/coupons/S700_3mm/cp{}.asc'.format(i))
        coupon.clean_initial()
        coupon.calc_init_stiffness()
        coupon.offset_to_0()
        coupon.add_initial_data(thicknesses[i-1], widths[i-1], l_0=l_0)
        coupon.calc_stress_strain()
        coupon.calc_young()
        coupon.calc_plastic_strain()
        coupon.calc_fy()

        cp["cp{}_3mm".format(i)] = coupon

    # 4 mm plate
    widths = [19.375, 20.335, 20.345]
    thicknesses = [4.0732, 4.0744, 4.0906]

    for i in [1, 2, 3]:
        coupon = CouponTest.from_file('./data/coupons/S700_4mm/cp{}.asc'.format(i))
        coupon.clean_initial()
        coupon.calc_init_stiffness()
        coupon.offset_to_0()
        coupon.add_initial_data(thicknesses[i-1], widths[i-1], l_0=l_0)
        coupon.calc_stress_strain()
        coupon.calc_young()
        coupon.calc_plastic_strain()
        coupon.calc_fy()

        cp["cp{}_4mm".format(i)] = coupon

    # 3 mm plate
    # for i in range(1, 7):
    #     coupon = CouponTest.from_file('./data/coupons/3mm/cp{}.asc'.format(i))
    #     coupon.clean_initial()
    #     coupon.calc_init_stiffness()
    #     coupon.offset_to_0()
    #
    #     cp["cp{}_2mm".format(i)] = coupon
    #
    #
    # widths = [20.408, 20.386, 20.397, 20.366, 20.35, 20.39]
    # thicknesses = [1.884, 1.891, 1.9, 1.88, 1.882, 1.878]
    # l_0s = [80., 80., 80., 80., 80., 80.]
    #
    # for i in range(1, 7):
    #     x = cp["cp{}_2mm".format(i)]
    #     x.add_initial_data(thicknesses[i-1], widths[i-1], l_0s[i-1])
    #     x.calc_stress_strain()
    #     x.calc_young()
    #     x.calc_plastic_strain()
    #     # x.plot_stressstrain_eng()
    #
    #     ax = plt.axes()
    #     x.plot_stressstrain_eng(ax=ax)
    #
    # ax.plot([0.002, 0.002 + 900 / 210000], [0, 900])


    return cp

