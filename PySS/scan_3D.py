# -*- coding: utf-8 -*-

"""
Module containing methods related to 3D scanning.

"""
import numpy as np
from PySS import analytic_geometry as ag
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as intrp


class Scan3D:
    def __init__(self, points_wcsys=None):
        self.points_wcsys = points_wcsys

    @classmethod
    def from_pickle(cls, fh):
        """
        Create object by loading points on the world csys from an stl.

        Parameters
        ----------
        fh : str
            Path to the stl file.

        Returns
        -------

        """
        return cls(points_wcsys=ag.Points3D.from_pickle(fh))


class FlatFace(Scan3D):
    """
    Subclass of the Scan3D class, specifically for flat faces.

    Used for the individual faces of the polygonal specimens.

    """

    def __init__(self, points_wcsys=None):
        self.points_lcsys = None
        self.face2ref_dist = None
        self.ref_plane = None
        self.regular_grid = None

        super().__init__(points_wcsys=points_wcsys)

    def local_from_world(self):
        """
        Add a local coordinate object based on the world object.

        Add :obj:`ag.Points3D` grid on the local coordinate system of the reference plane.
        The scanned points are projected on the reference plane and the projections are rotated to lie on xy-plane.
        """
        # # Check if face2ref exists (The check is performed with an if in the beginning of the method to avoid unnecessary
        # #  calculations)
        # if self.face2ref_dist is None:
        #     print('There is no face2ref. Try `calc_face2ref_dist`.')
        #     return
        #
        # # Find scanned points projections on the ref plane.
        # proj_swarm = self.project_swarm(self.ref_plane)

        # Z-axis unit vector.
        v1 = np.r_[0, 0, 1]

        # The normalised norm vector of the plane (which will be aligned to z axis)
        v2 = ag.unit_vector(self.ref_plane.plane_coeff[0:3])

        # Find the angle between the zz axis and the plane's normal vector, v2
        rot_ang = ag.angle_between(v1, v2)

        # Find the rotation axis.
        rot_ax = ag.unit_vector(np.r_[-v2[1], v2[0], 0])

        # Lay the projections on xy
        # xy_swarm  = proj_swarm.rotate_swarm(rot_ang, rot_ax)
        transformed = self.points_wcsys.rotate_swarm(rot_ang, rot_ax)
        transformed = transformed.translate_swarm(ag.Point3D.from_coordinates(0, 0, self.ref_plane.plane_coeff[3]))

        # Fit a line on the transformed data to get the direction of the laid down facet.
        # The direction vector is multiplied by the sign of the y-coordinate so that it is always on quadrants 1 and 2.
        dir_line = ag.Line3D.from_fitting(transformed.swarm)
        direction = np.sign(dir_line.parallel[1])*(dir_line.parallel[:2])

        # Calculate the angle of the laid down facet to the x axis.
        rot_ang2 = ag.angle_between(direction, [1, 0])

        # Rotate the swarm again, this time around the z axis, so that it is aligned with the x axis
        transformed = transformed.rotate_swarm(rot_ang2, [0, 0, 1])

        # Check the orientation of the transformed swarm: the base of the specimen is always at the origin. Where is the
        # head of the specimen? (is it around 700 or -700). If it faces negative, rotate by another 180 deg.
        transformed.calc_csl()
        if transformed.centre[0] < 0:
            transformed = transformed.rotate_swarm(np.pi, [0, 0, 1])

        # Translate the swarm so that the centre is on the origin. Then, the base of the specimen should be on the
        # negative and the head on the positive
        transformed.calc_csl()
        translate_vect = ag.Point3D.from_coordinates(-transformed.centre[0], -transformed.centre[1], 0)
        transformed = transformed.translate_swarm(translate_vect)

        self.points_lcsys = transformed
        self.points_lcsys.calc_csl()

    def fit_plane(self):
        """
        Fit a plane on the scanned data (WCS).

        The Plane3D object is assigned in the `self.ref_plane`. The fitted plane is returned using the
        analytic_geometry.lstsq_planar_fit with the optional argument lay_on_xy=True. See
        analytic_geometry.lstsq_planar_fit documentation.
        """
        self.ref_plane = ag.Plane3D.from_fitting(self.points_wcsys, lay_on_xy=True)

    def offset_face(self, offset, offset_points=False):
        """
        Offset the plane and (optionally) the scanned data points (WCS).

        Useful for translating translating the scanned surface to the mid line.

        :param offset:
        :param offset_points:
        :return:
        """
        self.ref_plane.offset_plane(offset)

        if offset_points:
            offsetted_points = self.points_wcsys.translate_swarm(ag.Point3D(self.ref_plane.plane_coeff[:3] * offset))
            self.points_wcsys.swarm = offsetted_points.swarm

    def calc_face2ref_dist(self):
        """Calculates distances from facet points to the reference plane."""
        if self.ref_plane:
            self.face2ref_dist = []
            for point in self.points_wcsys:
                self.face2ref_dist.append(point.distance_to_plane(self.ref_plane))

    def plot_face(self, fig=None, reduced=None):
        """
        Surface plotter.

        Plot the 3d points and the fitted plane.

        Parameters
        ----------
        fig : Object of class matplotlib.figure.Figure, optional
            The figure window to be used for plotting. By default, a new window is created.

        """

        #TODO: continue
        # Average and range of the points.
        self.points_wcsys.calc_csl()
        plot_dim = max(self.points_wcsys.size[0], self.points_wcsys.size[1], self.points_wcsys.size[2])

        # Get a figure to plot on
        if fig is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        else:
            ax = fig.get_axes()[0]

        # Plot scanned points
        if self.points_wcsys.swarm:
            print('Plotting scanned points')
            self.points_wcsys.plot_swarm(fig=fig, reduced=reduced)
        else:
            print('No scanned points to plot.')

        # Check if the the object contains a plane and plot it.
        if self.ref_plane:
            print('Plotting plane.')
            # Create a grid for for xy
            # get height and width (z and x axes) limits from points
            x_lims = [self.points_wcsys.centre[0] - self.points_wcsys.size[0] / 2.,
                      self.points_wcsys.centre[0] + self.points_wcsys.size[0] / 2.]
            y_lims = [self.points_wcsys.centre[1] - self.points_wcsys.size[1] / 2.,
                      self.points_wcsys.centre[1] + self.points_wcsys.size[1] / 2.]
            z_lims = [self.points_wcsys.centre[2] - self.points_wcsys.size[2] / 2.,
                      self.points_wcsys.centre[2] + self.points_wcsys.size[2] / 2.]

            ll1 = self.ref_plane.xy_return(z_lims[0])
            ll2 = self.ref_plane.xy_return(z_lims[1])

            if self.points_wcsys.size[0]>self.points_wcsys.size[1]:
                y1 = ll1.y_for_x(x_lims[0])
                y2 = ll1.y_for_x(x_lims[1])
                y3 = ll2.y_for_x(x_lims[0])
                y4 = ll2.y_for_x(x_lims[1])

                x = np.array([x_lims, x_lims])
                y = np.array([[y1, y2], [y3, y4]])
            else:
                x1 = ll1.x_for_y(y_lims[0])
                x2 = ll1.x_for_y(y_lims[1])
                x3 = ll2.x_for_y(y_lims[0])
                x4 = ll2.x_for_y(y_lims[1])

                x = np.array([[x1, x2], [x3, x4]])
                y = np.array([y_lims, y_lims])

            # x, y = np.meshgrid(x_lims, y_lims)
            # print(x, type(x))
            # print(y, type(y))
            # evaluate the plane function on the grid.
            z = self.ref_plane.z_return(x, y)
            # or expressed using matrix/vector product
            # z = np.dot(np.c_[xx, yy, np.ones(xx.shape)], self.plane_coeff).reshape(x.shape)

            # Plot the plane
            ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.2)

        else:
            print('No reference plane to plot. Use `fit_plane` to create one.')

        # Regulate figure.
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim3d(self.points_wcsys.centre[0] - plot_dim / 2, self.points_wcsys.centre[0] + plot_dim / 2)
        ax.set_ylim3d(self.points_wcsys.centre[1] - plot_dim / 2, self.points_wcsys.centre[1] + plot_dim / 2)
        ax.set_zlim3d(self.points_wcsys.centre[2] - plot_dim / 2, self.points_wcsys.centre[2] + plot_dim / 2)
        plt.show()

        # Return the figure handle.
        return fig

    def regulate_imperf(self):
        """
        Re-sample the list of imperfection displacements on a regular 2D grid.
        """
        # Re-mesh flat face local coordinates on a regular grid
        points = np.column_stack([self.points_lcsys.get_xs(), self.points_lcsys.get_ys()])
        values = self.points_lcsys.get_zs()
        l_x = self.points_lcsys.size[0]
        l_y = self.points_lcsys.size[1]
        n_tot = len(self.points_lcsys)
        n_x = np.ceil(np.sqrt(n_tot * l_x / l_y))
        n_y = np.ceil(np.sqrt(n_tot * l_y / l_x))

        step = min([l_x / n_x, l_y / n_y])

        grid_x, grid_y = np.meshgrid(
            np.arange(
                self.points_lcsys.lims[0][0],
                self.points_lcsys.lims[0][1],
                step=step),
            np.arange(
                self.points_lcsys.lims[1][0],
                self.points_lcsys.lims[1][1],
                step=step))

        grid_z = intrp.griddata(points, values, (grid_x, grid_y), method='nearest')

        self.regular_grid = [grid_x, grid_y, grid_z]

    def fft(self):
        """
        Perform 3D fourier
        """
        # Initialise an empty array
        field = self.regular_grid[2]

        # Number of samples on each direction
        n_x, n_y = field.shape[1], field.shape[0]

        # Perform 2D fourier and shift the result to centre
        freq = np.fft.fft2(field)

        # Calculate the magnitude and phase spectra. Keep only half symmetric results on each axis.
        magnitude_log_spectrum = np.log(np.abs(freq))[:n_y // 2, :n_x // 2]
        magnitude_spectrum = np.abs(freq)[:n_y // 2, :n_x // 2]
        phase_spectrum = np.angle(freq)[:n_y // 2, :n_x // 2]

        # Reconstruct the initial field
        #re_field = np.real(np.fft.ifft2(freq))

        # Max amp waves
        max_waves = []
        temp_mag_field = magnitude_spectrum[:5, :10]
        for i in range(10):
            cur_max = np.unravel_index(temp_mag_field.argmax(), temp_mag_field.shape)
            max_waves.append([cur_max, temp_mag_field[cur_max]])
            temp_mag_field[cur_max] = 0

        print("Max waves :")
        print(max_waves)

        # Plot
        fig = plt.figure()

        fig.add_subplot(411)
        plt.imshow(field, cmap='gray')
        plt.title('Field'), plt.xticks([]), plt.yticks([])
        plt.colorbar()

        fig.add_subplot(412)
        plt.imshow(magnitude_log_spectrum, cmap='gray')
        plt.title('Magnitude spectrum')

        fig.add_subplot(413)
        plt.imshow(phase_spectrum, cmap='gray')
        plt.title('Phase spectrum')

        fig.add_subplot(414)
        plt.imshow(magnitude_log_spectrum[:15, :15], cmap='gray')
        plt.title('Magnitude spectrum zoom')
        plt.colorbar()

        # surface plot
        mini_x, mini_y = np.meshgrid(
            np.arange(0, temp_mag_field.shape[1]),
            np.arange(0, temp_mag_field.shape[0]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(mini_x, mini_y, temp_mag_field)


class RoundedEdge(Scan3D):
    """
    A scanned rounded edge.

    """

    def __init__(self, points_wcsys=None):
        self.theoretical_edge = None
        self.edge_points = None
        self.circles = None
        self.edge2ref_dist = None
        self.ref_line = None
        self.fft_results = None
        self.u_max = None

        super().__init__(points_wcsys=points_wcsys)

    def add_theoretical_edge(self, line):
        """
        Add a reference line for the edge.

        Useful when the rounded edge lies between flat faces and the theoretical edge is at their intersection.

        Parameters
        ----------
        line : Line3D
            Theoretical edge line to be added. This line should be calculated as the intersection of the facets sharing
            this edge.
        """
        if isinstance(line, ag.Line3D):
            self.theoretical_edge = line
        else:
            print("ref_line must be Line3D")
            return NotImplemented

    def fit_circles(self, axis=None, offset=None):
        """
        Fit a series of circles along the length of the rounded edge.

        The scanned data are first grouped together based on their z-coordinate and then a horizontal circle is fitted
        for each group of points.

        Note
        ----
        The resulted circle from fitting at each height, z, are checked so that the centres are closer to the origin
        than the points that generated it, essentially checking if the curvature of the points is concave towards the
        origin. If not, the circle is ignored and a `None` placeholder is appended to the list of circles.

        """
        if axis is None:
            axis = 0

        if offset is None:
            offset = 0

        self.points_wcsys.quantize(axis=axis)
        self.circles = []
        for group in self.points_wcsys.grouped_data:
            circle = ag.Circle2D.from_fitting(group)
            if all([np.linalg.norm(i.coords[:2]) > np.linalg.norm(circle.centre) for i in group]):
                self.circles.append(circle)
                self.circles[-1].radius = self.circles[-1].radius + offset
            else:
                print('Suspicious circle from fitting ignored at height:    {}'.format(group[0].coords[2]))

    def calc_edge_points(self, other):
        """
        Intersect scanned points with a surface between the theoretical edge line and a given line.

        This function is used to find points on the scanned rounded corner. Circles are fitted on the
        scanned points on different positions. Then the circles are intersected with the theoretical edge and another
        line specified by the user (e.g. the centre of the column). A list of points is generated which represent the
        real edge of rounded corner.

        Parameters
        ----------
        other: Line3D object
            Secondary 3D line used together with the theoretical edge 3D line. The edge circles are intersected with
            line segments defined by points of those two 3D lines.

        """
        if isinstance(other, ag.Line3D):
            self.edge_points = []
            # Loop through the circles that represent the edge roundness at different heights.
            for circle in self.circles:
                # Get the z-coordinate (height) of the current point
                z_current = circle.points[0].coords[2]
                #print('Finding edge point at height {}'.format(z_current))

                # Get the x-y coordinates of the edge reference line and the mid-line for the given height, z.
                theoretical_line_point = self.theoretical_edge.xy_for_z(z_current)[:2]
                other_line_point = other.xy_for_z(z_current)

                # Create a temporary line object from the two points.
                intersection_line = ag.Line2D.from_2_points(theoretical_line_point, other_line_point[:2])

                # Intersect this temporary line with the current circle.
                line_circle_intersection = circle.intersect_with_line(intersection_line)

                # If the line does not intersect with the current circle, print on screen and continue.
                if line_circle_intersection is None:
                    print("Line and circle at height {} do not intersect. Point ignored.".format(z_current))

                else:
                    # If the line intersects with the circle, select the intersection point which is closest to the
                    # theoretical line.
                    dist = [np.linalg.norm(theoretical_line_point - x) for x in line_circle_intersection]
                    ref_point = line_circle_intersection[dist.index(min(dist))]

                    # Append the point to the list of edge_points
                    self.edge_points.append(ag.Point3D(np.append(ref_point, z_current)))
        else:
            print('The input object is not of the class `Line3D`')
            return NotImplemented

    def calc_ref_line(self):
        """
        Calculate the reference line.

        The reference line for the edge is defined as the best fit straight line to the edge points. For more
        information on the edge points, see the `intersect_data` method.
        """
        self.ref_line = ag.Line3D.from_fitting(self.edge_points)

    def calc_edge2ref_dist(self):
        """Calculate distances of edge points to the reference line."""
        if self.ref_line and self.ref_line is not NotImplemented:

            # Get the relative position of the first point on the reference edge line. This will be used as origin for
            # for the projected edge points on the reference line.
            origin = np.dot(self.ref_line.parallel, self.edge_points[0].coords)

            position = []
            distance = []

            for x in self.edge_points:
                # Find the distances from the the real edge and the ref line points to the (0, 0). Based on which one is
                # further away from the origin, the sign of the distance is assigned
                edge = np.linalg.norm(np.r_[0, 0] - x.coords[:2])
                refp = np.linalg.norm(np.r_[0, 0] - self.ref_line.xy_for_z(x.coords[2])[:2])
                s = np.sign(edge - refp)

                # calculate the distance of the edge point to the ref line and give this distance the sign calculated.
                distance.append(s * x.distance_to_line(self.ref_line))

                # calculate the position of the projected real edge point on the reference line, using as origin the
                # projection of the first point (see above).
                position.append(abs(origin - np.dot(self.ref_line.parallel, x.coords)))

            # assign positions and distances on the parent object
            self.edge2ref_dist = [position, distance]


        else:
            print('No reference line. First, add a reference line to the object. Check if the fitting process on the '
                  'edge points converged. Edge ignored.')
            return NotImplemented

    def plot_imp(self):
        if self.edge2ref_dist:
            plt.plot(self.edge2ref_dist[0], self.edge2ref_dist[1])
        else:
            print('No information for distances between edge points and reference line. Try the calc_edge2ref_dist '
                  'method.')

    #TODO: docstring
    def fft(self):
        """
        Perform fft on the edge points (imperfections).
        """
        # Create a linear space for equally distributed samples
        n_of_samples = len(self.edge2ref_dist[0])
        edge_length = self.edge2ref_dist[0][-1]
        t = np.linspace(0, edge_length, n_of_samples, endpoint=True)

        # Print information
        dt = t[1] - t[0]
        fa = 1.0 / dt  # scan frequency
        print('dt=%.5f mm (Sampling distance)' % dt)
        print('fa=%.2f samples/mm' % fa)

        # Displacement values (the signal)
        s = self.edge2ref_dist[1]

        # Perform fft without windowing
        Y = np.fft.fft(s)
        N = n_of_samples // 2 + 1

        # Frequency domain x-Axis with 'frequencies' up to Nyquist
        X = edge_length * np.linspace(0, fa / 2, N, endpoint=True)

        # Perform fft with windowing
        # Window functions: Choose one of the three
        hann = np.hanning(len(s))
        hamm = np.hamming(len(s))
        black = np.blackman(len(s))

        Yhann = np.fft.fft(hann * s)
        Yhamm = np.fft.fft(hamm * s)
        Yblack = np.fft.fft(black * s)

        # Plot all
        # plt.figure(figsize=(7, 3))
        #
        # plt.subplot(241)
        # plt.plot(t, s)
        # plt.title('No windowing')
        # plt.ylim(np.min(s) * 3, np.max(s) * 3)
        #
        # plt.subplot(242)
        # plt.plot(t, s * hann)
        # plt.title('Hanning')
        # plt.ylim(np.min(s) * 3, np.max(s) * 3)
        #
        # plt.subplot(243)
        # plt.plot(t, s * hamm)
        # plt.title('Hamming')
        # plt.ylim(np.min(s) * 3, np.max(s) * 3)
        #
        # plt.subplot(244)
        # plt.plot(t, s * black)
        # plt.title('Blackman')
        # plt.ylim(np.min(s) * 3, np.max(s) * 3)
        #
        # plt.subplot(245)
        # plt.bar(2 * X[:20], 2.0 * np.abs(Y[:20]) / N)
        # plt.xlabel('Length to buckle width ratio, l/w')
        #
        # plt.subplot(246)
        # plt.bar(2 * X[:20], 2.0 * np.abs(Yhann[:20]) / N)
        # plt.xlabel('Length to buckle width ratio, l/w')
        #
        # plt.subplot(247)
        # plt.bar(2 * X[:20], 2.0 * np.abs(Yhamm[:20]) / N)
        # plt.xlabel('Length to buckle width ratio, l/w')
        #
        # plt.subplot(248)
        # plt.bar(2 * X[:20], 2.0 * np.abs(Yblack[:20]) / N)
        # plt.xlabel('Length to buckle width ratio, l/w')

        self.fft_results = (2 * X[:20], 2.0 * np.abs(Yhann[:20]) / N)

        self.u_max = self.fft_results[1][1:].max() / (edge_length / self.fft_results[0][1 + self.fft_results[1][1:].argmax()])


def main():
    print('Module successfully loaded.')


if __name__ == "__main__":
    main()
