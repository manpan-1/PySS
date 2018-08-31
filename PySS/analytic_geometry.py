# -*- coding: utf-8 -*-

"""
Module for basic analytic geometry.

"""
import numpy as np
import pickle
from scipy import odr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
import os


class Plane3D:
    """
    Flat plane in three dimensions.

    A flat plane in the 3 dimensions is expressed in the implicit form, `a*x + b*y + c*z + d = 0`. The vector
    `v = [a, b, c, d]` is here called 'plane coefficients'. A Plane3D object can be created by setting the plane
    coefficients directly or by fitting on a set of 3D points using the alternative constructor `from_fitting`.

    Parameters
    ----------
    plane_coeff : {float, float, float, float}, optional
        Coefficients of the implicit form of the plane, `a*x + b*y + c*z + d = 0`.

    """

    def __init__(self, plane_coeff=None):
        self.plane_coeff = plane_coeff

    def __and__(self, other):
        """
        Ampersand operator returns the intersection line.

        The ampersand operator between two Plane3D objects calculates their intersection, constructs and returns a
        Line3D object.

        Parameters
        ----------
        other : :obj:`Plane3D`
            The plane to intersect with.

        Returns
        -------
        Line3D
            The intersection line of the two planes.

        """

        if isinstance(other, Plane3D):
            # Calculate the parallel vector of the intersection line as the dot product of the vectors normal to the
            # planes.
            parallel = np.cross(np.r_[self.plane_coeff[0], self.plane_coeff[1], self.plane_coeff[2]],
                                np.r_[other.plane_coeff[0], other.plane_coeff[1], self.plane_coeff[2]])
            # Normalise the direction vector.
            parallel = unit_vector(parallel)

            # Calculate the intersection of the line with the xy plane
            a_1, a_2 = self.plane_coeff[0], other.plane_coeff[0]
            b_1, b_2 = self.plane_coeff[1], other.plane_coeff[1]
            c_1, c_2 = self.plane_coeff[2], other.plane_coeff[2]
            d_1, d_2 = self.plane_coeff[3], other.plane_coeff[3]
            # y_0 = (c_2 * a_1 - c_1 * a_2) / (b_1 * a_2 - b_2 * a_1)
            # x_0 = (-b_1 / a_1) / y_0 - c_1/a_1
            y_0 = (d_2 * a_1 - d_1 * a_2) / (b_1 * a_2 - b_2 * a_1)
            x_0 = (b_1 * y_0 + d_1) / (-a_1)
            z_0 = 0
            p_0 = np.array([x_0, y_0, z_0])

            return Line3D.from_point_and_parallel(p_0, parallel)
        else:
            return NotImplemented

    def offset_plane(self, offset):
        # TODO: Conclude on the proper way to document input arguments of specific form, e.g. 3by1 array of floats...
        """
        Offset the plane.

        Useful for translating translating a scanned surface to the mid line by offsetting half the thickness.

        Parameters
        ----------
        offset : float
            Offset distance.

        Returns
        -------
        plane_coeff : (3,) ndarray
            The method modifies the `plane_coeff` to apply the offset.

        """
        self.plane_coeff = np.append(self.plane_coeff[:3], self.plane_coeff[3] - offset)

    def z_return(self, x, y):
        """
        Calculate z of a plane for given x, y.

        Parameters
        ----------
        x : float or numpy.ndarray
        y : float or numpy.ndarray

        Returns
        -------
        float or numpy.ndarray

        """
        if not isinstance(self.plane_coeff, np.ndarray):
            print('Wrong or missing plane coefficients')
            return NotImplemented
        alpha = (-self.plane_coeff / self.plane_coeff[2])
        z = alpha[0] * x + alpha[1] * y + alpha[3]
        return z

    def xy_return(self, z0):
        """
        Intersect with a z = z0 plane.

        The intersection of the plane with a vertical plane z = z0 is calculated and a Line2D object is constructed and
        returned. Note that the returned Line2D object is unaware of the hight z0 that was used to create it and it is
        only an expression of x and y.

        Parameters
        ----------
        z0 : float

        Returns
        -------
        Line2D object
            The intersection line of the plane with the z=z0 plane.

        """
        return Line2D.from_line_coeff(
            self.plane_coeff[0],
            self.plane_coeff[1],
            self.plane_coeff[2] * z0 + self.plane_coeff[3]
        )

    @classmethod
    def from_fitting(cls, points, lay_on_xy=False):
        # TODO: fix the "array like" the form of the array...
        """
        Create plane object from fitting on points.

        A simple least squares fit is performed on the given points and a Plane2D object is constructed. Additionally,
        the points can be rotated to be nearly horizontal and the least squares fitting is performed on the rotated
        position. The resulted plane is then rotated back to the initial position of the points. This methodology helps
        when the points are nearly vertical.

        Parameters
        ----------
        points : [[x0, y0, z0], ..., [xn, yn, zn]] array like
            The points for which the fitting is performed

        lay_on_xy : bool, optional
            Perform a single least squares (False) or rotate the points to be nearly horizontal and then perform the
            least squares fitting (True). Default value is `False`.

        Returns
        -------
        Plane3D

        Notes
        -----
        When the `lay_on_xy` flag is true, the fitting is no longer a simple least squares and it rather resembles an
        orthogonal distance regression. The set of given points is rotated based on a least squares on their initial
        position so when the least squares is performed again, this time on the rotated position, the distances of the
        points on the z-axis is almost perpendicular to the resulted plane.

        See Also
        --------
        lstsq_planar_fit : fit plane on data

        """
        plane_coeff = lstsq_planar_fit(points, lay_on_xy=lay_on_xy)
        return cls(plane_coeff=plane_coeff)

    @classmethod
    def from_coefficients(cls, a, b, c, d):
        if all([isnumber(a), isnumber(b), isnumber(c), isnumber(d)]):
            return cls(np.r_[a, b, c, d])
        else:
            print("At least one of the input objects is not numeric")


class Circle2D:
    # TODO: [x,y] floats on the docstring.
    """
    A circle in two dimensions.

    A circle is expressed in the form `(x - x0)^2 + (y - y0)^ = r^2` where p = [x0, y0] is the centre and r the radius.
    A circle2D object can be created either by setting th radius and centre directly or by best fit on a set of given
    points, using the `from_fitting` class method. The intersection points of the circle with a given line (if any) can
    be calculated using the `intersect_with_line` method.

    Parameters
    ----------
    radius : float, optional
    centre : [float, float], optional

    """

    def __init__(self, radius=None, centre=None):
        self.radius = radius
        self.centre = centre
        self.points = None

    def intersect_with_line(self, line):
        # TODO: what happens if not implementid (no intersection) on docstring.
        """
        Intersect circle with line.

        Parameters
        ----------
        line : Line2D
            Line to intersect the circle with.

        Returns
        -------
        list of [x, y, z] points
            The intersection points. If the line and the circle do not intersect, no return is given.

        """
        if isinstance(line, Line2D):
            a, b, c = line.line_coeff
            xc, yc = self.centre
            radius = self.radius

            # Terms of the quadratic equation for the line-circle intersection.
            alfa = 1 + (a / b) ** 2
            beta = 2 * (c * a / b ** 2 + yc * a / b - xc)
            gama = xc ** 2 + yc ** 2 - radius ** 2 + (c / b) ** 2 + 2 * c * yc / b

            # Solution
            x_intersect = solve_quadratic(alfa, beta, gama)

            # If no intersection, exit and return None
            if x_intersect is None:
                print("The line does not intersect with the circle")
                return

            # TODO: Here, a Point2D class would be convenient. Implement in analytic_geometry
            # Calculate y.
            y_intersect = np.r_[-(c + a * x_intersect[0]) / b, -(c + a * x_intersect[1]) / b]

            # Points.
            point1 = np.r_[x_intersect[0], y_intersect[0]]
            point2 = np.r_[x_intersect[1], y_intersect[1]]
            return [point1, point2]
        else:
            print("The input object is not of the class `Line2D`")
            return NotImplemented

    def plot_circle(self):
        """
        Plot circle

        Points, best fit circle and center are plotted on a new figure.

        """

        plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
        plt.axis('equal')

        theta_fit = np.linspace(-np.pi, np.pi, 180)

        x_fit3 = self.centre[0] + self.radius * np.cos(theta_fit)
        y_fit3 = self.centre[1] + self.radius * np.sin(theta_fit)
        plt.plot(x_fit3, y_fit3, 'r-.', label='odr fit', lw=2)

        plt.plot([self.centre[0]], [self.centre[1]], 'kD', mec='w', mew=1)

        # draw
        plt.xlabel('x')
        plt.ylabel('y')

        plt.draw()
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        vmin = min(xmin, ymin)
        vmax = max(xmax, ymax)

        #TODO: Fix error: implement Point3D object
        # plot data
        plt.plot(self.points[:, 0], self.points[:, 1], 'ro', label='data', ms=8, mec='b', mew=1)
        plt.legend(loc='best', labelspacing=0.1)

        plt.xlim(xmin=vmin, xmax=vmax)
        plt.ylim(ymin=vmin, ymax=vmax)

        plt.grid()
        plt.title('Least Squares Circle')

    @classmethod
    def from_fitting(cls, points):
        # TODO: fix points docstring...
        """
        Create circle object from fitting on points.

        Parameters
        ----------
        points : [float, float]
            Points for which a best-fit circle is sought. The method makes sense for more than 3 points.

        Returns
        -------
        Circle2D

        See Also
        --------
        circular_fit : fit circle on data

        """
        xc, yc, rad = circular_fit(points)
        obj = cls(radius=rad, centre=np.r_[xc, yc])
        obj.points = points
        return obj


class Line3D:
    # TODO: create xz_for_y and yz_for_x methods, equivalent to xy_for_z AND fix the parameters' docstring
    """
    A line in three dimensions.

    A line in the 3 dimensions is expressed by it's direction vector and a point belonging to the line. A Line3D object
    can be constructed in the following ways:
    - From a given point and parallel, `from_point_and_parallel`
    - From 2 given points, `from_2_points`
    - From 2 points loaded from a pickle file, `from_pickle`

    Parameters
    ----------
    point : [float, float, float], optional
    parallel : [float, float, float], optional

    Notes
    -----
    Even though an instance of the class can be created directly by giving a point and a parallel vector directly, it is
    preferred to use the dedicated class method :obj:`from_point_and_parallel` because it automatically normalises the
    direction vector.

    """

    def __init__(self, point=None, parallel=None):
        self.point = point
        self.parallel = parallel

    @classmethod
    def from_point_and_parallel(cls, point, parallel):
        """
        Create line object from a point and a parallel vector.

        Parameters
        ----------
        point : array like
        parallel : array like

        Returns
        -------
        Line3D

        """
        # Normalise the given parallel vector
        parallel = unit_vector(np.r_[parallel])
        return cls(point=Point3D(np.r_[point]), parallel=parallel)

    @classmethod
    def from_2_points(cls, point1, point2):
        """
        Create line object from 2 points.

        Parameters
        ----------
        point1, point2 : array like

        Returns
        -------
        Line3D

        """
        point1 = np.r_[point1]
        point2 = np.r_[point2]
        # Calculate and normalise the direction vector.
        parallel = unit_vector(point1 - point2)
        return cls(point=Point3D(point1), parallel=parallel)

    @classmethod
    def from_pickle(cls, fh):
        """
        Import line from pickle.

        Used to import center lines for the polygonal specimens, as exported from blender.

        Parameters
        ----------
        fh: string
            Path and filename of the pickle file.
        """
        with open(fh, 'rb') as f:
            points = pickle.load(f)

        return cls.from_2_points(np.r_[points[0]], np.r_[points[1]])

    @classmethod
    def from_fitting(cls, points):
        """
        Create a `Line3D` object from fitting on points

        Parameters
        ----------
        points : list of Point3D objects
            Points to which the 3D line is fitted

        """
        # TODO: here the input is checked only to be a list. Check also that the contents are Point3D objects.
        if isinstance(points, list):
            linepts = line3d_fit(points)
            if not linepts is NotImplemented:
                return cls.from_2_points(linepts[0], linepts[1])
            else:
                print('Line did not converge. A `NotImplemented` object is returned.')
                return NotImplemented
        else:
            print("The input object is not of the class a list of points.")
            return NotImplemented

    def xy_for_z(self, z_1):
        """Return x, y for a given z"""
        t = (z_1 - self.point.coords[2]) / self.parallel[2]
        x_1 = self.parallel[0] * t + self.point.coords[0]
        y_1 = self.parallel[1] * t + self.point.coords[1]
        return np.r_[x_1, y_1, z_1]

    def plot_line(self, ends=None, fig=None):
        """
        Line segment plotter.

        Plot a segment of the line between two values of the parameter `t` in x=x0 + a*t

        Parameters
        ----------
        ends : array like, optional
            The end values for the parametric form of the line segment to be plotted (array like with 2 values). Default
            is [-1, 1]
        fig : Object of class matplotlib.figure.Figure, optional
            The figure window to be used for plotting. By default, a new window is created.

        Returns
        -------
        figure handle

        """
        if ends is None:
            ends = np.array([-1, 1])

        x = self.point.coords[0] + self.parallel[0] * np.r_[ends]
        y = self.point.coords[1] + self.parallel[1] * np.r_[ends]
        z = self.point.coords[2] + self.parallel[2] * np.r_[ends]

        if fig is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            ax = fig.get_axes()[0]
        ax.plot(x, y, z)

        plt.show()

        return fig

    def intersect_with_plane(self, plane):
        """
        Intersect line with a given plane.

        Parameters
        ----------
        plane: :obj:`Plane3D` object
            Plane with which the line is intersected.
        Returns
        -------
        :obj:`Point3D` object

        """
        # Calculate the parameter value t0 for which the line coincides with the plane.
        try:
            numerator = np.dot(np.append(self.point.coords, [1]), plane.plane_coeff)
            denominator = np.dot(self.parallel, plane.plane_coeff[:3])
        except TypeError as ex:
            print(ex)
            print('Plane3D object must be given.')
            return NotImplemented

        t0 = - numerator / denominator

        # Calculate intersection point coordinates.
        x, y, z = self.parallel * t0 + self.point.coords

        # Create and return a Point3D object
        return Point3D.from_coordinates(x, y, z)


class Line2D:
    # TODO: fix list items in docstring.
    """
    A line in two dimensions.

    Line objects on the 2 dimensions are expressed by a point and a parallel vector, or/and by the coefficients
    (a, b, c) of the implicit form `a*x + b*y + c = 0`. There are 3 different ways to create a Line2D object:
    -- From 2 points,
    -- from point and parallel vector,
    -- from the expression coefficients, given as a vector [a, b, c]
    To create a Line2D object, use one of the four `from_...` class methods.

    Parameters
    ----------
    point : ndarray [x0, y0], optional
        Point belonging to the line. Default is `None`
    parallel : ndarray [v_x, v_y], optional
        2D vector parallel to the line. Default is `None`
    line_coeff : ndarray [a, b, c], optional
        Coefficients of the implicit form of the line, `a*x + b*y + c = 0`.

    """

    def __init__(self, point=None, parallel=None, line_coeff=None):
        self.point = point
        self.parallel = parallel
        self.line_coeff = line_coeff

    @classmethod
    def from_point_and_parallel(cls, point, parallel):
        """
        Create line object from point and parallel.

        Parameters
        ----------
        point : array like
        parallel : array like

        Returns
        -------
        Line2D

        """
        # Normalise the given parallel vector
        parallel = unit_vector(np.r_[parallel])
        line_coeff = np.r_[-parallel[1], parallel[0], parallel[1] * point[0] - parallel[0] * point[1]]
        return cls(point=np.r_[point], parallel=parallel, line_coeff=line_coeff)

    @classmethod
    def from_2_points(cls, point1, point2):
        """
        Create line object from 2 points.

        Parameters
        ----------
        point1, point2 : array like

        Returns
        -------
        Line2D

        """
        point1 = np.r_[point1]
        point2 = np.r_[point2]
        # Calculate and normalise the direction vector.
        parallel = unit_vector(point1 - point2)
        return cls.from_point_and_parallel(point1, parallel)

    @classmethod
    def from_line_coeff(cls, alfa, beta, gama):
        """
        Create a line from the coefficients of the form `a*x + b*y + c = 0`.

        Parameters
        ----------
        alfa, beta, gama : float

        Returns
        -------
        Line2D

        """
        parallel = np.r_[beta, -alfa]
        point = [0, -(gama / beta)]
        line = cls.from_point_and_parallel(point, parallel)
        line.line_coeff = np.r_[alfa, beta, gama]
        return line

    @classmethod
    def from_pickle(cls, fh):
        """
        Import line from pickle.

        Parameters
        ----------
        fh: string
            Path and filename of the pickle file.

        Returns
        -------
        Line2D

        """
        with open(fh, 'rb') as f:
            points = pickle.load(f)
            return cls.from_2_points(np.r_[points[0]], np.r_[points[1]])

    def x_for_y(self, y):
        """
        Return x a given y.

        Parameters
        ----------
        y : float

        Returns
        -------
        x : float

        """

        return (-self.line_coeff[1] * y - self.line_coeff[2]) / self.line_coeff[0]

    def y_for_x(self, x):
        """
        Return y a given x.

        Parameters
        ----------
        x : float

        Returns
        -------
        y : float

        """

        return (-self.line_coeff[0] * x - self.line_coeff[2]) / self.line_coeff[1]

    def plot_line(self, ends=None, fig=None):
        # TODO: Check if the method works fine and revise the docstring.
        """
        Line segment plotter.

        Plot a segment of the line between two values of the parameter `t`  x=x0 + a*t

        Parameters
        ----------
        ends : array like, optional
            The end values for the parametric form of the line segment to be plotted (array like with 2 values). Default
            is [-1, 1]
        fig : Object of class matplotlib.figure.Figure, optional
            The figure window to be used for plotting. By default, a new window is created.

        Returns
        -------
        figure handle

        """
        if ends is None:
            ends = np.array([-1, 1])

        x = self.point[0] + self.parallel[0] * np.r_[ends]
        y = self.point[1] + self.parallel[1] * np.r_[ends]

        if fig is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.get_axes()[0]
        ax.plot(x, y, label='parametric curve')
        ax.legend()

        plt.show()

        return fig


class Point3D:
    """A point in 3 dimensions."""

    def __init__(self, coords):
        self.coords = coords

    def __add__(self, other):
        return Point3D(self.coords + other.coords)

    @classmethod
    def from_coordinates(cls, x, y, z):
        """
        Create a Point3D by from it's 3 coordinates.

        Parameters
        ----------
        x, y, z : float
            The values for the three coordinates.

        """
        if isnumber(x) and isnumber(y) and isnumber(z):
            return cls(np.r_[x, y, z])
        else:
            print('At least one of the inputs is not numeric.')
            return NotImplemented

    def distance_to_plane(self, plane):
        """
        Measure the distance of the point to a plane.

        Parameters
        ----------
        plane : Plane3D object.
            Plane from to which the distance is measured.

        Returns
        -------
        float

        """
        # Check if the given input is of the class Plane3D
        if isinstance(plane, Plane3D):
            dist = np.dot(plane.plane_coeff, np.concatenate((self.coords, [1]))) / np.linalg.norm(plane.plane_coeff[:3])
            return dist
        else:
            print('The input object is not of the class `Plane3D`')
            return NotImplemented

    def distance_to_line(self, line):
        """
        Measure the distance of the point to a line.

        Parameters
        ----------
        line : Line3D object.
            Line from to which the distance is measured.

        Returns
        -------
        float

        """
        if isinstance(line, Line3D):
            s_direction = line.parallel
            s_norm = np.linalg.norm(s_direction)
            p_diff = line.point.coords - self.coords
            d = np.linalg.norm(np.cross(p_diff, s_direction)) / s_norm
            return d
        else:
            print('The input object is not of the class `Line3D`')
            return NotImplemented

    def project_on_plane(self, plane):
        """
        Project the point on a given plane.

        Parameters
        ----------
        plane: :obj:`Plane3D` object
            Plane on which the point is projected.
        Returns
        -------
        :obj:`Point3D` object

        """
        # Create a 3D line that passes from the point and is orthogonal to the plane.
        try:
            proj_line = Line3D.from_point_and_parallel(self.coords, plane.plane_coeff[:3])
        except TypeError as ex:
            print(ex)
            print('Given input must be Plane3D.')
            return NotImplemented

        # Intersect the line with the plane and return the point
        return proj_line.intersect_with_plane(plane)

    def rotate_point(self, rot_ang, rot_ax):
        """
        Rotate points for given angle around axis.

        Parameters
        ----------
        points : 2d array like
            List of points[[x0, y0, z0], ...[xn, yn, zn]]
        rot_ang : float
            Rotation angle in rads.
        rot_ax : array like
            Vector u = [xu, yu, zu] of the axis around which the points are rotated.

        Returns
        -------
        ndarray
            List of points[[x0, y0, z0], ...[xn, yn, zn]]

        """
        # Rotation matrix
        sint = np.sin(rot_ang)
        cost = np.cos(rot_ang)
        ux, uy, uz = rot_ax

        rot_mtx = np.r_[
            [[cost + ux ** 2 * (1 - cost), ux * uy * (1 - cost) - uz * sint, ux * uz * (1 - cost) + uy * sint],
             [uy * ux * (1 - cost) + uz * sint, cost + uy ** 2 * (1 - cost), uy * uz * (1 - cost) - ux * sint],
             [uz * ux * (1 - cost) - uy * sint, uz * uy * (1 - cost) + ux * sint, cost + uz ** 2 * (1 - cost)]]
        ]

        # Transform the points.
        return Point3D(np.dot(self.coords, rot_mtx))


class Points3D:
    """
    Swarm of 3D points.

    """
    def __init__(self, swarm=None):
        self.swarm = swarm
        self.grouped_data = None
        self.centre = None
        self.size = None
        self.lims = None

    def __iter__(self):
        return iter(self.swarm)

    def __len__(self):
        return len(self.swarm)

    @classmethod
    def from_stl_file(cls, fh, del_original=None):
        """
        Import stl file.

        Alternative constructor, creates a Points3D object by reading data from an .stl file. In case the file is created
        by Creaform's software (it is detected using name of the solid object as described in it's frist line), it is
        corrected accordingly before importing. The original file is renamed by adding '_old' before the extension or
        they can be deleted automatically if specified so.

        Parameters
        ----------
        fh : str
            File path.
        del_original : bool, optional
            Keep or delete the original file. Default is keep.
        """
        with open(fh, 'r') as f:
            fl = f.readlines(1)[0]
            identifier = fl.split(None, 1)[1]

        if identifier == 'ASCII STL file generated with VxScan by Creaform.\n':
            # Repair the file format
            Points3D.repair_stl_file_structure(fh, del_original=del_original)

        return cls(swarm=Points3D.array2points(mesh.Mesh.from_file(fh)))

    @classmethod
    def from_pickle(cls, fh):
        """
        Method for importing a pickle file containing x, y, z, coordinates.

        Used to import data exported from blender. The pickle file is should contain a list of lists.

        """
        with open(fh, 'rb') as fh:
            return cls(swarm=Points3D.array2points(np.array(pickle.load(fh))))

    @classmethod
    def from_coordinates_file(cls, fh):
        """
        Method reading text files containing x, y, z coordinates.

        Used to import data from 3D scanning files.
        """

        # Open the requested file.
        with open(fh, 'r') as f:
            # Number of points.
            n_of_points = len(f.readlines())

            # Initialise a numpy array for the values.
            swarm = np.empty([n_of_points, 3])

            # Reset the file read cursor and loop over the lines of the file populating the numpy array.
            f.seek(0)
            for i, l in enumerate(f):
                swarm[i] = l.split()

        return cls(swarm=Points3D.array2points(swarm))

    @staticmethod
    def repair_stl_file_structure(fh, del_original=None):
        """
        Repair header-footer of files created by Creaform's package.

        The .stl files created by Creaform's software are missing standard .stl header and footer. This method will
        create a copy of the requested file with proper header-footer using the filename (without the extension) as a
        name of the solid.

        Parameters
        ----------
        fh : str
            File path.
        del_original : bool, optional
            Keep or delete the original file. Default is keep.
        """
        if del_original is None:
            del_original = False
        solid_name = os.path.splitext(os.path.basename(fh))[0]

        start_line = "solid " + solid_name + "\n"
        end_line = "endsolid " + solid_name
        old_file = os.path.splitext(fh)[0] + 'old.stl'

        os.rename(fh, old_file)
        with open(old_file) as fin:
            lines = fin.readlines()
        lines[0] = start_line
        lines.append(end_line)

        with open(fh, 'w') as fout:
            for line in lines:
                fout.write(line)

        if del_original:
            os.remove(old_file)

    @staticmethod
    def array2points(array):
        """
        Convert an array of coordinates to a list of Point3D objects.

        Parameters
        ----------
        array : {n*3} np.ndarray

        Returns
        -------
        list of Point3D.

        """
        if isinstance(array, np.ndarray):
            if np.shape(array)[1] == 3:
                point_list = []
                for i in array:
                    point_list.append(Point3D.from_coordinates(i[0], i[1], i[2]))
                return point_list
            else:
                print('Wrong array dimensions. The array must have 3 columns.')
                return NotImplemented
        else:
            print('Wrong input. Input must be np.ndarray')
            return NotImplemented

    def sort_on_axis(self, axis=None):
        """
        Sort scanned data.

        The scanned points are sorted for a given axis.

        Parameters
        ----------
        axis : {0, 1, 2}, optional
            Axis for which the points are sorted. 0 for `x`, 1 for `y` and 2 for `z`.
            Default is 0

        """
        if axis is None:
            axis = 0

        self.swarm.sort(key=lambda x: x.coords[axis])

    def quantize(self, axis=None, tolerance=None):
        """
        Group the scanned data.

        The points with difference on a given axis smaller than the tolerance are grouped together and stored in a list
        in the attribute `grouped_data`.

        Parameters
        ----------
        axis : {0, 1, 2}, optional
            Axis for which the points are grouped. 0 for `x`, 1 for `y` and 2 for `z`.
            Default is 0.
        tolerance : float
            Distance tolerance for grouping the points.

        """
        if axis is None:
            axis = 0

        if tolerance is None:
            tolerance = 1e-4

        self.sort_on_axis(axis=axis)
        self.grouped_data = [[self.swarm[0]]]
        for point in self.swarm:
            if abs(point.coords[axis] - self.grouped_data[-1][0].coords[axis]) < tolerance:
                self.grouped_data[-1].append(point)
            else:
                self.grouped_data.append([point])

    def calc_csl(self):
        """
        Calculate centre, size and limits of the swarm.

        The results are assigned to the equivalent attributes self.centre, self.size, self.lims.

        """
        # Bounding box of the points.
        x_min = min([i.coords[0] for i in self.swarm])
        x_max = max([i.coords[0] for i in self.swarm])
        y_min = min([i.coords[1] for i in self.swarm])
        y_max = max([i.coords[1] for i in self.swarm])
        z_min = min([i.coords[2] for i in self.swarm])
        z_max = max([i.coords[2] for i in self.swarm])
        x_range = abs(x_max - x_min)
        y_range = abs(y_max - y_min)
        z_range = abs(z_max - z_min)
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_min + z_max) / 2

        self.centre = np.r_[x_mid, y_mid, z_mid]
        self.size = np.r_[x_range, y_range, z_range]
        self.lims = np.r_[[[x_min, x_max], [y_min, y_max], [z_min, z_max]]]

    def translate_swarm(self, vect):
        """
        Translate the swarm by a given vector point.

        Parameters
        ----------
        vect : :obj:`Point3D`
            Translation vector.

        """
        return Points3D(swarm=[p + vect for p in self])

    def rotate_swarm(self, rot_ang, rot_ax):
        """
        Rotate points for given angle around axis.

        Parameters
        ----------
        rot_ang : float
            Rotation angle in rads.
        rot_ax : array like
            Vector u = [xu, yu, zu] of the axis around which the points are rotated.

        Returns
        -------
        :obj:`Points3D`
            Rotated point swarm.

        """
        # Transform the points.
        return Points3D(swarm=[p.rotate_point(rot_ang, rot_ax) for p in self])

    def project_swarm(self, plane):
        """
        Project swarm of points on a given plane.

        Parameters
        ----------
        plane : :obj:`Plane3D`

        Returns
        -------
        :obj:`Points3D`

        """
        proj_swarm = []
        for x in self:
            proj_swarm.append(x.project_on_plane(plane))

        return Points3D(swarm=proj_swarm)

    def plot_swarm(self, fig=None, reduced=None):
        """
        Method plotting the model as a 3D surface.

        Parameters
        ----------
        fig : Object of class matplotlib.figure.Figure, optional
            The figure window to be used for plotting. By default, a new window is created.
        reduced: float, optional
            A reduced randomly selected subset of points is plotted (in case the data is too dense for plotting). The
            reduced size is given as a ratio of the total number of points, e.g `reduced=0.5` plots half the points. By
            default, all points are plotted.

        """
        # Get a figure to plot on
        if fig is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        else:
            ax = fig.get_axes()[0]

        # Make a randomly selected subset of points acc. to the input arg 'reduced=x'.
        if isinstance(reduced, float) and (0 < reduced < 1):
            n = list(np.random.choice(
                len(self.swarm),
                size=round(len(self.swarm) * reduced),
                replace=False
            ))
        else:
            n = range(0, len(self.swarm))

        # Create the x, y, z lists
        x, y, z = [], [], []
        for i in n:
            x.append(self.swarm[i].coords[0])
            y.append(self.swarm[i].coords[1])
            z.append(self.swarm[i].coords[2])

        # Plot the data
        ax.scatter(x, y, z, c="black", s=1)

    def get_xs(self):
        """Return an array of the `x` coordinates."""
        xs = []
        for i in self:
            xs.append(i.coords[0])

        return np.array(xs)

    def get_ys(self):
        """Return an array of the `y` coordinates."""
        ys = []
        for i in self:
            ys.append(i.coords[1])

        return np.array(ys)

    def get_zs(self):
        """Return an array of the `z` coordinates."""
        zs = []
        for i in self:
            zs.append(i.coords[2])

        return np.array(zs)


def lstsq(points):
    # TODO: Description of the return value
    """
    Perform a least squares fit and return the plane coefficients of the form 'a*x + b*y + c*z + d = 0'.
    The return vector beta=[a, b, c, d] is normalised for the direction v=[a, b, c] to be a unit vector.

    Parameters
    ----------
    points : :obj:`Points3D` object

    Returns
    -------
    ndarray

    """
    # best-fit linear plane
    x_list = [i.coords[0] for i in points]
    y_list = [i.coords[1] for i in points]
    z_list = [i.coords[2] for i in points]
    a = np.c_[x_list, y_list, np.ones(len(points))]
    c, _, _, _ = np.linalg.lstsq(a, z_list)  # coefficients

    # The coefficients are returned as an array beta=[a, b, c, d] from the implicit form 'a*x + b*y + c*z + d = 0'.
    # The vector is normalized so that [a, b, c] has a unit length and `d` is positive.
    return np.r_[c[0], c[1], -1, c[2]] / (np.linalg.norm([c[0], c[1], -1]) * np.sign(c[2]))


# TODO: Something fishy is going on in the lstsq_planar_fit: all points seem to be on the same side of the result plane.
def lstsq_planar_fit(points, lay_on_xy=False):
    """
    Fit a plane to 3d points.

    A regular least squares fit is performed. If the argument lay_on_xy is given True, a regular least squares
    fitting is performed and the result is used to rotate the points so that they come parallel to the xy-plane.
    Then, a second least squares fit is performed and the result is rotated back to the initial position. This
    procedure helps overcoming problems with near-vertical planes.

    Parameters
    ----------
    points : :obj:`Points3D`
        List of points[[x0, y0, z0], ...[xn, yn, zn]]
    lay_on_xy : bool, optional
        If True, the fitting is performed after the points are rotated to lay parallel to the xy-plane based on an
        initial slope estimation. Default is False, which implies a single last squares on the points on their initial
        position.

    Returns
    -------
    ndarray
        Vector with the plane coefficients, beta=[a, b, c, d] from the implicit form 'a*x + b*y + c*z + d = 0'

    """
    if lay_on_xy is None:
        lay_on_xy = False

    # Pseudo-orthogonal fit
    if lay_on_xy:
        # Perform least squares fit on the points "as is"
        beta1 = lstsq(points)

        # Z-axis unit vector.
        v1 = np.r_[0, 0, 1]

        # The normalised norm vector of the plane (which will be aligned to z axis)
        v2 = unit_vector(beta1[0:3])

        # Find the angle between the zz axis and the plane's normal vector, v2
        rot_ang = angle_between(v1, v2)

        # Find the rotation axis.
        rot_ax = unit_vector(np.r_[-v2[1], v2[0], 0])

        # Transform the points so that v2 is aligned to z.
        transformed = points.rotate_swarm(rot_ang, rot_ax)

        # Perform least squares.
        beta2 = lstsq(transformed)

        # Return the fitted plane to the original position of the points.
        beta2[:3] = Point3D(beta2[:3]).rotate_point(-rot_ang, rot_ax).coords

        # Return the plane coefficients.
        return beta2

    else:
        # Perform a least squares fitting directly on the original position and return the coefficients.
        return lstsq(points)


# TODO: adjust method to the Point3D update or delete.
def quadratic_fit(points):
    """
    Fit a quadratic surface to 3D points.

    A regular least squares fit is performed (no error assumed in the given z-values).

    Parameters
    ----------
    points : list of [x, y, z] points

    Returns
    -------
    ndarray

    """
    # best-fit quadratic curve
    a = np.c_[
        np.ones(points.shape[0]),
        points[:, :2],
        np.prod(points[:, :2], axis=1),
        points[:, :2] ** 2]

    beta, _, _, _ = np.linalg.lstsq(a, points[:, 2])
    return beta


# TODO: Function not used. Decide what to do with it.
def odr_planar_fit(points, rand_3_estimate=False):
    """
    Fit a plane to 3d points.

    Orthogonal distance regression is performed using the odrpack.

    Parameters
    ----------
    points : list of [x, y, z] points
    rand_3_estimate : bool, optional
        First estimation of the plane using 3 random points from the input points list.
        Default is False which implies a regular least square fit for the first estimation.

    Returns
    -------
    ndarray

    """

    def f_3(beta, xyz):
        """ implicit definition of the plane"""
        return beta[0] * xyz[0] + beta[1] * xyz[1] + beta[2] * xyz[2] + beta[3]

    # # Coordinates of the 2D points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # x = np.r_[9, 35, -13, 10, 23, 0]
    # y = np.r_[34, 10, 6, -14, 27, -10]
    # z = np.r_[100, 101, 101, 100, 101, 101]

    if rand_3_estimate:
        # initial guess for parameters
        # select 3 random points
        i = np.random.choice(len(x), size=3, replace=False)

        # Form the 3 points
        r_point_1 = np.r_[x[i[0]], y[i[0]], z[i[0]]]
        r_point_2 = np.r_[x[i[1]], y[i[1]], z[i[1]]]
        r_point_3 = np.r_[x[i[2]], y[i[2]], z[i[2]]]

        # Two vectors on the plane
        v_1 = r_point_1 - r_point_2
        v_2 = r_point_1 - r_point_3

        # normal to the 3-point-plane
        u_1 = np.cross(v_1, v_2)

        # Construct the first estimation, beta0
        d_0 = u_1[0] * r_point_1[0] + u_1[1] * r_point_1[1] + u_1[2] * r_point_1[2]
        beta0 = np.r_[u_1[0], u_1[1], u_1[2], d_0]
    else:
        beta0 = lstsq_planar_fit(points)

    # Create the data object for the odr. The equation is given in the implicit form 'a*x + b*y + c*z + d = 0' and
    # beta=[a, b, c, d] (beta is the vector to be fitted). The positional argument y=1 means that the dimensionality
    # of the fitting is 1.
    lsc_data = odr.Data(np.row_stack([x, y, z]), y=1)
    # Create the odr model
    lsc_model = odr.Model(f_3, implicit=True)
    # Create the odr object based on the data, the model and the first estimation vector.
    lsc_odr = odr.ODR(lsc_data, lsc_model, beta0)
    # run the regression.
    lsc_out = lsc_odr.run()

    return lsc_out.beta / lsc_out.beta[3]


# TODO: recheck the docstring, changes are made to adopt the Point3D objects.
def circular_fit(points):
    """
    Fit a circle to a set of 2D points.

    The fitting is performed using the ODR from scipy.

    Parameters
    ----------
    points : list
        List of points[[x0, y0, z0], ...[xn, yn, zn]]

    Returns
    -------
    ndarray
        Vector with the plane coefficients, beta=[xc, yc, r] of the circle from the implicit definition of the circle
        `(x - xc) ^ 2 + (y - yc) ^ 2 - r ^ 2`.

    Notes
    -----
    The code is partly taken from the scipy Cookbook.

    https://github.com/mpastell/SciPy-CookBook/blob/master/originals/Least_Squares_Circle_attachments/least_squares_circle_v1d.py
    """
    x = np.r_[[i.coords[0] for i in points]]
    y = np.r_[[i.coords[1] for i in points]]

    def calc_r(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_3b(beta, var):
        """ implicit definition of the circle """
        return (var[0] - beta[0]) ** 2 + (var[1] - beta[1]) ** 2 - beta[2] ** 2

    def jacb(beta, var):
        """
        Jacobian function with respect to the parameters beta.

        Returns
        -------
        df_3b/dbeta

        """
        xc, yc, r = beta
        xi, yi = var

        df_db = np.empty((beta.size, var.shape[1]))
        df_db[0] = 2 * (xc - xi)  # d_f/dxc
        df_db[1] = 2 * (yc - yi)  # d_f/dyc
        df_db[2] = -2 * r  # d_f/dr

        return df_db

    def jacd(beta, var):
        """
        Jacobian function with respect to the input x.

        Returns
        -------
        df_3b/dx

        """
        xc, yc, r = beta
        xi, yi = var

        df_dx = np.empty_like(var)
        df_dx[0] = 2 * (xi - xc)  # d_f/dxi
        df_dx[1] = 2 * (yi - yc)  # d_f/dyi

        return df_dx

    def calc_estimate(data):
        """Return a first estimation on the parameter from the data."""
        xc0, yc0 = data.x.mean(axis=1)
        r0 = np.sqrt((data.x[0] - xc0) ** 2 + (data.x[1] - yc0) ** 2).mean()
        return xc0, yc0, r0

    # for implicit function :
    #       data.x contains both coordinates of the points
    #       data.y is the dimensionality of the response
    lsc_data = odr.Data(np.row_stack([x, y]), y=1)
    lsc_model = odr.Model(f_3b, implicit=True, estimate=calc_estimate, fjacd=jacd, fjacb=jacb)
    lsc_odr = odr.ODR(lsc_data, lsc_model)  # beta0 has been replaced by an estimate function
    lsc_odr.set_job(deriv=3)  # use user derivatives function without checking
    lsc_out = lsc_odr.run()

    xc_odr, yc_odr, r_odr = lsc_out.beta
    # ri_3b = calc_r(xc_odr, yc_odr)
    # residu_3b = sum((ri_3b - r_odr) ** 2)
    # residu2_3b = sum((ri_3b ** 2 - r_odr ** 2) ** 2)

    return xc_odr, yc_odr, r_odr


def line3d_fit(points):
    """
    Fit a line in 3D on a list of points

    Parameters
    ----------
    points : list of Point3D
        Cloud of points to best fit the line.

    Returns
    -------
    Line3D
        Fitted line.

    """
    x, y, z = [], [], []
    for i in points:
        x.append(i.coords[0])
        y.append(i.coords[1])
        z.append(i.coords[2])

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    data = np.concatenate((x[:, np.newaxis],
                           y[:, np.newaxis],
                           z[:, np.newaxis]),
                          axis=1)

    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = data.mean(axis=0)

    # Do an SVD on the mean-centered data.
    try:
        uu, dd, vv = np.linalg.svd(data - datamean)
    except:
        print('Fitting did not converge.')
        return NotImplemented

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    # Generate 2 points along this best fit line.
    linepts = vv[0] * np.mgrid[-100:100:2j][:, np.newaxis]

    linepts += datamean
    return linepts

    # TODO: use the code below to make a plotting function.
    # # shift by the mean to get the line in the right place
    # # Verify that everything looks right.
    #
    # import matplotlib.pyplot as plt
    # import mpl_toolkits.mplot3d as m3d
    #
    # ax = m3d.Axes3D(plt.figure())
    # ax.scatter3D(*data.T)
    # ax.plot3D(*linepts.T)
    # plt.show()


def unit_vector(vector):
    """ Returns the unit vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def solve_quadratic(a, b, c):
    """
    Solve a quadratic equation for real roots.

    Parameters
    ----------
    a, b, c : float
        Coefficients of the form `a * x ^ 2 + b * x + c = 0`

    Returns
    -------
    list of float
        The solution [x1, x2] for discriminant >= 0.
        A `None` type is returned if the equation has no real roots.

    """
    # calculate the discriminant
    d = (b ** 2) - (4 * a * c)

    if d < 0:
        print('No real solutions.')
        return
    else:
        # find two solutions
        x1 = (-b - np.sqrt(d)) / (2 * a)
        x2 = (-b + np.sqrt(d)) / (2 * a)
        return [x1, x2]


def isnumber(x):
    """
    Checks if the given object is numeric.

    Either `float` or `int` will return `True`. In any other case the return is `False`.

    Parameters
    ----------
    x : any
        Object to be checked.

    Returns
    -------
    boolean.

    """
    return isinstance(x, int) or isinstance(x, float)
