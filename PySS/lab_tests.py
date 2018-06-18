# -*- coding: utf-8 -*-

"""
Module with functions related to laboratory work and data acquired with CATMAN.

"""

import numpy as np
import csv
import codecs
from os.path import basename, splitext
from matplotlib import pyplot as plt


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

    def plot2d(self, x_data, y_data, ax=None, scale=None):
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

        """

        if ax is None:
            fig = plt.figure()
            plt.plot()
            ax = fig.axes[0]
        elif not isinstance(ax, type(plt.axes())):
            print('Unexpected input type. Input argument `ax` must be of type `matplotlib.pyplot.axes()`')
            return NotImplemented
        
        if scale is None:
            scale=(1, 1)

        ax.plot(self.channels[x_data]["data"]*scale[0], self.channels[y_data]["data"]*scale[1], label=self.name)

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

        # Find the indices of the values larger then 1 kN, as a way of excluding head and tail recorded data.
        valid = np.where(self.channels["Load"]["data"] > 1)[0]

        self.channels["Load"]["data"] = self.channels["Load"]["data"][valid]
        self.channels["Epsilon"]["data"] = self.channels["Epsilon"]["data"][valid]

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

    def plot_stressstrain_eng(self, ax=None):
        ax = self.plot2d('StrainEng', 'StressEng', ax=ax)
        plt.xlabel('Strain, $\epsilon_{eng}$ [%]')
        plt.ylabel('Stress, $\sigma_{eng}$ [MPa]')
        plt.title(self.name)
        return ax


def main():
    cp = []
    for i in range(1, 7):
        coupon = CouponTest.from_file('./data/coupons/cp{}.asc'.format(i))
        coupon.clean_initial()
        coupon.calc_init_stiffness()
        coupon.offset_to_0()

        cp.append(coupon)


    widths = [20.408, 20.386, 20.397, 20.366, 20.35, 20.39]
    thicknesses = [1.884, 1.891, 1.9, 1.88, 1.882, 1.878]
    l_0s = [80., 80., 80., 80., 80., 80.]

    for i , x in enumerate(cp):
        x.add_initial_data(thicknesses[i], widths[i], l_0s[i])
        x.calc_stress_strain()
        x.calc_young()
        x.calc_plastic_strain()
        # x.plot_stressstrain_eng()

        ax = plt.axes()
        x.plot_stressstrain_eng(ax=ax)

    ax.plot([0.002, 0.002 + 900 / 210000], [0, 900])

    return cp

