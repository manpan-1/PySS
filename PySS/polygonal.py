# -*- coding: utf-8 -*-

"""
A framework for the study of polygonal profiles.

"""
import os
import numpy as np
import PySS.steel_design as sd
import PySS.lab_tests as lt
import PySS.analytic_geometry as ag
import PySS.scan_3D as s3d
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PolygonalColumn:
    """
    Polygonal column.

    """

    def __init__(self, name=None, theoretical_specimen=None, real_specimen=None, experiment_data=None):
        self.name = name
        self.theoretical_specimen = theoretical_specimen
        self.real_specimen = real_specimen
        self.experiment_data = experiment_data

    def add_theoretical_specimen(self,
                                 n_sides,
                                 length,
                                 f_yield,
                                 fab_class,
                                 r_circle=None,
                                 p_class=None,
                                 thickness=None
                                 ):

        if [i is None for i in [r_circle, p_class, thickness]].count(True) > 1:
            print('Not enough info. Two out of the three optional arguments {r_circle, p_class, thickness}'
                  ' must be given.')
            return
        else:
            if p_class is None:
                self.theoretical_specimen = TheoreticalSpecimen.from_geometry(
                    n_sides,
                    r_circle,
                    thickness,
                    length,
                    f_yield,
                    fab_class
                )
            elif r_circle is None:
                self.theoretical_specimen = TheoreticalSpecimen.from_slenderness_and_thickness(
                    n_sides,
                    p_class,
                    thickness,
                    length,
                    f_yield,
                    fab_class
                )
            else:
                self.theoretical_specimen = TheoreticalSpecimen.from_slenderness_and_radius(
                    n_sides,
                    r_circle,
                    p_class,
                    length,
                    f_yield,
                    fab_class
                )

    def add_real_specimen(self, path):
        """
        Add data from scanning pickle file.

        Adds a self.specimen object of the RealSpecimen class. Scanned data are loaded from a list of pickle files
        corresponding to sides and edges, each file containing point coordinates. The pickle files are assumed to follow
        the the filename structure:

        files with points of sides: `side_XX.pkl`
        files iwth points od edges: `edge_XX.pkl`
        where `XX` is an ascending number starting from 01.

        Parameters
        ----------
        path : str
            Path containing side_XX.pkl and edge_XX.pkl files. eg `/home/user/` (terminating backslash required).

        """
        # Scan the given path for filenames.
        file_list = os.listdir(path)

        n_sides, n_edges = 0, 0
        side_filename_numbers, edge_filename_numbers = [], []
        for fname in file_list:
            if fname[-4:] == '.pkl':
                if fname[:5] == 'side_':
                    n_sides = n_sides + 1
                    side_filename_numbers.append(int(fname[5:7]))
                if fname[:5] == 'edge_':
                    n_edges = n_edges + 1
                    edge_filename_numbers.append(int(fname[5:7]))

        if n_sides == 0 and n_edges == 0:
            print('No side or edge files were found in the directory.')
            return NotImplemented

        # Sort the numbers fetched from the filenames and check if they are sequential or if there are numbers missing.
        side_filename_numbers.sort()
        edge_filename_numbers.sort()
        if (not all([x == num - 1 for x, num in enumerate(side_filename_numbers)]) and
           not all([x == num - 1 for x, num in enumerate(edge_filename_numbers)])):
            print("Problem with filenames. Check if the filenames are correct (see method's documentation) ant the "
                  "numbering in the filenames is sequential (no sides or edges missing)")
            return NotImplemented

        # Create a polygon specimen object.
        if self.theoretical_specimen is None:
            print('No theoretical specimen defined. Before adding data of the real scanned specimen, it is necessary '
                  'to create the corresponding theoretical specimen.')
            return
        else:
            specimen = RealSpecimen(thickness=self.theoretical_specimen.geometry.thickness)

        # Add a center line for the specimen.
        # TODO: add real centreline from file.
        print('Adding centre-line from pickle.')
        specimen.centre_line_from_pickle(path + 'centreline.pkl')

        # Add all sides and edges.
        # they consist of FlatFace and RoundedEdge instances.
        specimen.add_all_sides(n_sides, path + 'side_', fit_planes=True, offset_to_midline=True)
        # Check if the existing edges found in the directory correspond one by one to the sides. If so, then the
        # intersection lines of adjacent sides are calculated and added to the edges as reference lines. Otherwise, the
        # edges are imported from whatever files are found and no reference lines are calculated.
        if side_filename_numbers == edge_filename_numbers:
            intrsct_lines = True
        else:
            intrsct_lines = False

        specimen.add_all_edges(n_edges, path + 'edge_', intrsct_lines=intrsct_lines)
        # Find a series of points for each edge based on the scanned surface.
        specimen.find_real_edges(offset_to_midline=True, ref_lines=True)

        # Calculate the initial imperfection displacements based on the edge and facet reference line and plane
        # accordingly.
        specimen.find_edge_imperfection_displacements()
        specimen.find_facet_imperfection_displacements()

        # Extract the maximum imperfection displacement from each facet and edge.
        specimen.gather_max_imperfections()

        # Assign the constructed specimen to the object
        self.real_specimen = specimen

    def add_experiment(self, fh):
        """Add and post-process data from a test"""
        self.experiment_data = TestData.from_file(fh)
        self.experiment_data.specimen_length = self.theoretical_specimen.geometry.length
        self.experiment_data.cs_area = self.theoretical_specimen.cs_props.area
        self.experiment_data.process_data()

    def report_real_specimen(self):
        """Print a report for the processed scanned data of the real specimen."""
        print('Report for {}'.format(self.name))
        self.real_specimen.print_report()


class TheoreticalSpecimen(sd.Part):
    """
    Properties and calculations of a theoretical (ideal geometry) polygonal column.

    """

    def __init__(self,
                 geometry=None,
                 cs_props=None,
                 material=None,
                 struct_props=None,
                 bc_loads=None):

        super().__init__(
            geometry,
            cs_props,
            material,
            struct_props,
            bc_loads)

    @classmethod
    def from_geometry(
            cls,
            n_sides,
            r_circle,
            thickness,
            length,
            f_yield,
            fab_class
    ):
        """
        Create theoretical polygonal column object for given geometric data.

        The constructor calculates properties of the polygonal column object (cross-section props,
        resistance, geometric props etc). The calculated data is then used to construct an object.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.

        r_circle : float
            Radius of the circle circumscribed to the polygon.

        thickness : float
            Thickness of the cross-section.

        length : float
            Length of the column.

        f_yield : float
            Yield stress of the material.

        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.

        """

        # Create material
        material = sd.Material(210000, 0.3, f_yield)
        epsilon = np.sqrt(235. / f_yield)

        # Radius of the polygon's circumscribed circle
        r_circum = (np.pi * r_circle) / (n_sides * np.sin(np.pi / n_sides))

        # Diameter
        diam_circum = 2 * r_circum

        # Central angles
        theta = 2 * np.pi / n_sides

        # Width of each side
        facet_width = diam_circum * np.sin(np.pi / n_sides)

        # Polar coordinate of the polygon vertices on the cross-section plane
        phii = []
        for i_index in range(n_sides):
            phii.append(i_index * theta)

        # Polygon corners coordinates.
        x_corners = tuple(r_circum * np.cos(phii))
        y_corners = tuple(r_circum * np.sin(phii))

        # Cross-sectional properties
        nodes = [x_corners, y_corners]
        elem = [
            list(range(0, len(x_corners))),
            list(range(1, len(x_corners))) + [0],
            len(x_corners) * [thickness]
        ]

        cs_sketch = sd.CsSketch(nodes, elem)
        geometry = sd.Geometry(cs_sketch, length, thickness)

        # Additional geometric properties (exclusive to the polygonal)
        geometry.r_circle = r_circle
        geometry.facet_width = facet_width
        geometry.n_sides = n_sides

        cs_props = sd.CsProps.from_cs_sketch(cs_sketch)
        cs_props.max_dist = r_circum
        cs_props.min_dist = np.sqrt(r_circum ** 2 - (facet_width / 2) ** 2)

        lmbda_y = sd.lmbda_flex(
            length,
            cs_props.area,
            cs_props.moi_1,
            kapa_bc=1.,
            e_modulus=material.e_modulus,
            f_yield=material.f_yield
        )

        lmbda_z = sd.lmbda_flex(
            length,
            cs_props.area,
            cs_props.moi_2,
            kapa_bc=1.,
            e_modulus=material.e_modulus,
            f_yield=material.f_yield
        )

        # Axial compression resistance , Npl
        n_pl_rd = n_sides * sd.n_pl_rd(thickness, facet_width, f_yield)

        # Compression resistance of equivalent cylindrical shell
        n_b_rd_shell = 2 * np.pi * r_circle * thickness * sd.sigma_x_rd(
            thickness,
            r_circle,
            length,
            f_yield,
            fab_quality=fab_class,
            gamma_m1=1.
        )

        # Plate classification acc. to EC3-1-1
        p_classification = facet_width / (epsilon * thickness)

        # Tube classification slenderness acc. to EC3-1-1
        t_classification = 2 * r_circle / (epsilon ** 2 * thickness)

        struct_props = sd.StructProps(
            t_classification=t_classification,
            p_classification=p_classification,
            lmbda_y=lmbda_y,
            lmbda_z=lmbda_z,
            n_pl_rd=n_pl_rd,
            n_b_rd_shell=n_b_rd_shell
        )


        return cls(geometry, cs_props, material, struct_props)

    @classmethod
    def from_slenderness_and_thickness(
            cls,
            n_sides,
            p_classification,
            thickness,
            length,
            f_yield,
            fab_class
    ):
        """
        Create theoretical polygonal column object for given number of sides and cross-section slenderness.

        The constructor calculates properties of the polygonal column object (cross-section props,
        resistance, geometric props etc) which are then used to construct an object.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.

        p_classification : float
            Facet slenderness, c/(ε*t).

        thickness : float
            Thickness of the cross-section.

        length : float
            Length of the column.

        f_yield : float
            Yield stress of the material.

        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.

        """

        # Epsilon for the material
        epsilon = np.sqrt(235. / f_yield)

        # Radius of the equal perimeter cylinder
        r_circle = n_sides * thickness * epsilon * p_classification / (2 * np.pi)

        return cls.from_geometry(
            n_sides,
            r_circle,
            thickness,
            length,
            f_yield,
            fab_class
        )

    @classmethod
    def from_slenderness_and_radius(
            cls,
            n_sides,
            r_circle,
            p_classification,
            length,
            f_yield,
            fab_class
    ):
        """
        Create theoretical polygonal column object for given geometric data.

        The constructor calculates properties of the polygonal column object (cross-section props,
        resistance, geometric props etc). The calculated data is then used to construct an object.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.

        r_circle : float
            Radius of the circle circumscribed to the polygon.

        p_classification : float
            Facet slenderness, c/(ε*t).

        length : float
            Length of the column.

        f_yield : float
            Yield stress of the material.

        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.

        """
        # Epsilon for the material
        epsilon = np.sqrt(235. / f_yield)

        # Calculate the thickness
        thickness = 2 * np.pi * r_circle / (n_sides * epsilon * p_classification)

        return cls.from_geometry(
            n_sides,
            r_circle,
            thickness,
            length,
            f_yield,
            fab_class
        )


class RealSpecimen:
    """
    A column specimen of polygonal cross-section.

    Used for the scanned polygonal specimens.

    """

    def __init__(self, sides=None, edges=None, centre_line=None, thickness=None):
        if sides is None:
            sides = []

        if edges is None:
            edges = []

        self.sides = sides
        self.edges = edges
        self.centre_line = centre_line
        self.thickness = thickness
        self.max_edge_imp = None
        self.max_face_imp = None

    def centre_line_from_pickle(self, fh):
        """
        Import a centre-line to the polygonal object from a pickle file.

        The pickle file is expected to contain a list of 2 points from which the line is constructed. This method is
        used in combination with the equivalent `export` method from blender.

        Parameters
        ----------
        fh : str
            Path and filename of the pickle file.

        """
        self.centre_line = ag.Line3D.from_pickle(fh)

    def add_single_side_from_pickle(self, filename):
        """
        Create a FlatFace instance as one side af the polygon column.

        The FlatFace instance is created from a pickle file of scanned data points.

        :param filename:
        :return:
        """
        self.sides.append(s3d.FlatFace.from_pickle(filename))

    def add_all_sides(self, n_sides, prefix, fit_planes=False, offset_to_midline=False):
        """
        Add multiple sides.

        Multiple FlatFace instances are created as sides of the polygonal column. A series of files containing scanned
        data points must be given. The files should be on the same path and have a filename structure as:
        `path/basenameXX.pkl`, where XX is an id number in ascending order starting from 01.
        Only the `path/basename` is given as input to this method.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygonal cross-section to look for in the directory
        prefix : str
            Path and file name prefix for the pickle files containing the scanned data points.
        fit_planes :
            Perform least square fitting on the imported data to calculate the reference planes.
        offset_to_midline :
            Offset the data points and the fitted plane by half the thickness to be on the midline of the cross-section.

        """

        self.sides = []
        for x in range(1, n_sides + 1):
            print('Adding scanned data, facet:   {}'.format(x))
            self.sides.append(s3d.FlatFace.from_pickle(prefix + '{:02d}.pkl'.format(x)))

        if fit_planes:
            for i, x in enumerate(self.sides):
                print('Fitting a reference plane, facet:    {}'.format(i + 1))
                x.fit_plane()

        if offset_to_midline:
            offset = self.thickness / 2
            for i, x in enumerate(self.sides):
                print('Offsetting plane and points, facet:    {}'.format(i + 1))
                x.offset_face(offset, offset_points=True)

    def add_single_edge_from_pickle(self, filename):
        """
        Create a RoundEdge instance as one edges af the polygon column.

        The RoundEdge instance is created from a pickle file of scanned data points.

        :param filename:
        :return:
        """
        self.edges.append(s3d.RoundedEdge.from_pickle(filename))

    def add_all_edges(self, n_sides, prefix, intrsct_lines=False):
        """
        Add multiple edges.

        Multiple RoundEdge instances are created as edges of the polygonal column. A series of files containing scanned
        data points must be given. The files should be on the same path and have a filename structure as:
        `path/basenameXX.pkl`, where XX is an id number in ascending order starting from 01.
        Only the `path/filename` is given as input to this method.

        After adding the sequential edges, if intrsct_lines=True, the reference lines are calculated as the
        intersections of sequential sides.

        Parameters
        ----------
        n_sides : int
            Number of edges to be added (number of cross-sections sides).
        prefix : str
            Path and prefix of naming (read description for the expected naming scheme)
        intrsct_lines : bool
            Assign intersection lines to the edges from the intersection of adjacent facets.

        """
        self.edges = []
        for x in range(1, n_sides + 1):
            print('Adding scanned data, edge:    {}'.format(x))
            self.edges.append(s3d.RoundedEdge.from_pickle(prefix + '{:02d}.pkl'.format(x)))

        if intrsct_lines:
            for x in range(-len(self.sides), 0):
                print('Adding theoretical edge, edge:    {}'.format(x + n_sides + 1))
                self.edges[x].theoretical_edge = (self.sides[x].ref_plane & self.sides[x + 1].ref_plane)

    def find_real_edges(self, offset_to_midline=False, ref_lines=False):
        """
        Find edge points on the scanned rounded edge.

        A series of points is returned which represent the real edge of the polygonal column. Each point is calculated
        as  the intersection of a circle and a line at different heights of the column, where the circle is best fit to
        the rounded edge scanned points and the line passing through the reference edge (see `add_all_edges`
        documentation) and the polygon's centre line.

        Parameters
        ----------
        offset_to_midline : bool
            Offset the calculated points to the midline of the section based on the thickness property of the object.
        ref_lines : bool
            Assign reference lines to the edges by best fitting on the real edge points.

        """
        if offset_to_midline:
            offset = -self.thickness / 2
        else:
            offset = 0

        if isinstance(self.centre_line, ag.Line3D) and isinstance(self.edges, list):
            for i, x in enumerate(self.edges):
                print('Fitting circles and calculating edge points, edge:    {}'.format(i + 1))
                x.fit_circles(axis=2, offset=offset)
                x.calc_edge_points(self.centre_line)
        else:
            print('Wrong type inputs. Check if the real_specimen object has a centre line assigned to it and if it has'
                  'a list of edge lines.')
            return NotImplemented

        if ref_lines:
            for i, x in enumerate(self.edges):
                print('Calculating reference line by fitting on the edge points, edge:    {}'.format(i + 1))
                x.calc_ref_line()

    def find_edge_imperfection_displacements(self):
        """Calculate distances of edge points to each reference line."""
        for i, x in enumerate(self.edges):
            print('Calculating initial imperfection displacements, edge:    {}.'.format(i + 1))
            if x.ref_line:
                if x.ref_line is NotImplemented:
                    print('The reference line is type `NotImplemented`, fitting possibly did not converge.')
                    x.edge2ref_dist = NotImplemented
                else:
                    x.calc_edge2ref_dist()
            else:
                print('No reference line. Edge imperfection not calculated.')
                x.edge2ref_dist = NotImplemented

    def find_facet_imperfection_displacements(self):
        """Calculate distances of edge points to each reference line."""
        for i, x in enumerate(self.sides):
            print('Calculating initial imperfection displacements, facet:    {}'.format(i + 1))
            x.calc_face2ref_dist()

    def plot_all(self):
        """
        Plot all data.

        """
        max_z = max([x.scanned_data[:, 2].max() for x in self.sides])
        min_z = min([x.scanned_data[:, 2].min() for x in self.sides])
        fig1 = plt.figure()
        Axes3D(fig1)
        for i in range(-len(self.sides), 0):
            self.sides[i].plot_face(reduced=0.001, fig=fig1)
        for i in range(-len(self.edges), 0):
            self.edges[i].facet_intrsct_line.plot_line(fig=fig1, ends=[min_z, max_z])

    def gather_max_imperfections(self):
        """
        Collect initial imperfection info from all the edges and facets.

        """
        self.max_face_imp = []
        self.max_edge_imp = []
        for x in self.sides:
            self.max_face_imp.append(max(np.abs(x.face2ref_dist)))
        for x in self.edges:
            try:
                self.max_edge_imp.append(max(np.abs(x.edge2ref_dist)))
            except:
                self.max_edge_imp.append(NotImplemented)

    def print_report(self):
        """
        Print a report for the polygon column.

        """
        for i, x in enumerate(self.sides):
            if x.face2ref_dist is NotImplemented:
                print('No initial displacement data, facet: {}'.format(i + 1))
            else:
                print('Max init displacement from ref plane, facet: {}'.format(i + 1), max(np.abs(x.face2ref_dist)))

        for i, x in enumerate(self.edges):
            if x.edge2ref_dist is NotImplemented:
                print('No initial displacement data, edge: {}'.format(i + 1))
            else:
                print('Max init displacement from ref line, edge: {}'.format(i + 1),
                      max(np.abs(x.edge2ref_dist)))

        # TODO: Fix the following code and add more to the report.
        # max_z = max([x.scanned_data[:, 2].max() for x in self.sides])
        # min_z = min([x.scanned_data[:, 2].min() for x in self.sides])
        # for i in range(len(self.sides)):
        #     print('Side {} is : {}'.format(i + 1, self.sides[i].ref_plane.plane_coeff))
        #     print('')
        #     print('Edge {} (sides {}-{})\n    Direction : {}\n    Through points : \n{}\n{}'.format(
        #         i + 1,
        #         i + 1,
        #         i + 2,
        #         self.edges[i].facet_intrsct_line.parallel,
        #         self.edges[i].facet_intrsct_line.xy_for_z(min_z),
        #         self.edges[i].facet_intrsct_line.xy_for_z(max_z))
        #     )
        #     print('')


class TestData(lt.Experiment):
    def __init__(self, name=None, data=None, specimen_length=None, cs_area=None):
        self.specimen_length = specimen_length
        self.cs_area = cs_area

        super().__init__(name=name, data=data)

    def process_data(self):
        """

        :return:
        """
        self.calc_avg_strain()
        self.calc_disp_from_strain()
        self.calc_avg_stress()

    def add_eccentricity(self, axis, column, moi, min_dist, thickness, young):
        """
        Calculate eccentricity.

        Adds a column in the data dictionary for the eccentricity of the load application on a given axis based on
        two opposite strain measurements.
        """

        self.data['e_' + axis] = []
        for load, strain1, strain2 in zip(self.data['Load'], self.data[column[0]], self.data[column[1]]):
            self.data['e_' + axis].append(self.eccentricity_from_strain(
                load * 1000,
                [strain1 * 1e-6, strain2 * 1e-6],
                moi,
                min_dist + thickness / 2,
                young)
            )

    def offset_stroke(self, offset=None):
        """
        Offset stroke values.

        Parameters
        ----------
        offset : float, optional
            Distance to offset. By default, the initial displacement (first value) is used, effectively displaceing
            the values to start from 0.

        """
        if offset is None:
            offset = self.data['Stroke'][0]

        self.data['Stroke'] = self.data['Stroke'] - offset

    def calc_disp_from_strain(self):
        """Calculate the specimen clear axial deformation based on measured strains"""
        self.add_new_channel_zeros('disp_clear')
        self.data['disp_clear'] = self.data['avg_strain'] * self.specimen_length

    def calc_avg_strain(self):
        """Calculate the average strain from all strain gauges."""
        # Create new data channel.
        self.add_new_channel_zeros('avg_strain')
        i = 0
        # Collect all strain gauge records.
        for key in self.data.keys():
            if len(key) > 2:
                if key[:2].isdigit() and (key[2] is 'F') or (key[2] is 'C'):
                    self.data['avg_strain'] = self.data['avg_strain'] + self.data[key]
                    i += 1

        self.data['avg_strain'] = self.data['avg_strain'] / (i * 1e6)

    def calc_avg_stress(self):
        """Calculate the average stress based on the measured reaction force on the load cell and the
        theoretical area."""
        # Create new data channel.
        self.add_new_channel_zeros('avg_stress')
        self.data['avg_stress'] = self.data['Load'] * 1e3 / self.cs_area

    def plot_stroke_load(self, ax=None):
        """Load vs stroke curve plotter"""
        if ax is None:
            fig = plt.figure()
            plt.plot()
            ax = fig.axes[0]
            self.plot2d('Stroke', 'Load', ax=ax)
            ax.invert_xaxis()
            ax.invert_yaxis()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1])
            ax.set_xlabel('Displacement, u [mm]')
            ax.set_ylabel('Reaction, N [kN]')
            ax.grid()

            return ax
        elif not isinstance(ax, type(plt.axes())):
            print('Unexpected input type. Input argument `ax` must be of type `matplotlib.pyplot.axes()`')
            return NotImplemented
        else:
            self.plot2d('Stroke', 'Load', ax=ax)
            return ax

    def plot_strain_stress(self, ax=None):
        """Plot average strain vs average stress."""
        if ax is None:
            fig = plt.figure()
            plt.plot()
            ax = fig.axes[0]
            self.plot2d('avg_strain', 'avg_stress', ax=ax)
            ax.invert_xaxis()
            ax.invert_yaxis()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            ax.set_xlabel('Strain, ε')
            ax.set_ylabel('Stress, σ [Mpa]')
            ax.grid()

            return ax
        elif not isinstance(ax, type(plt.axes())):
            print('Unexpected input type. Input argument `ax` must be of type `matplotlib.pyplot.axes()`')
            return NotImplemented
        else:
            self.plot2d('avg_strain', 'avg_stress', ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            return ax

    def plot_disp_load(self, ax=None):
        """Plot load vs real displacement."""
        if ax is None:
            fig = plt.figure()
            plt.plot()
            ax = fig.axes[0]
            self.plot2d('disp_clear', 'Load', ax=ax)
            ax.invert_xaxis()
            ax.invert_yaxis()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1])
            ax.set_xlabel('Displacement, u [mm]')
            ax.set_ylabel('Reaction, N [kN]')
            ax.grid()

            return ax
        elif not isinstance(ax, type(plt.axes())):
            print('Unexpected input type. Input argument `ax` must be of type `matplotlib.pyplot.axes()`')
            return NotImplemented
        else:
            self.plot2d('disp_clear', 'Load', ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1])
            return ax

    @staticmethod
    def eccentricity_from_strain(load, strain, moi, dist, young=None):
        """
        Load eccentricity based on strain pairs.

        Calculate the eccentricity of an axial load to the neutral axis of a specimen for which pairs of strains are
        monitored with strain gauges. The eccentricity is calculated on one axis and requires the moment of inertia
        around it and a pair of strains on tow positions symmetric to the neutral axis. Elastic behaviour is assumed.

        """

        # Default values.
        if young is None:
            young = 210000.
        else:
            young = float(young)

        # Eccentricity.
        ecc = (strain[0] - strain[1]) * young * moi / (2 * load * dist)

        # Return
        return ecc


def semi_closed_polygon(n_sides, radius, t, tg, rbend, nbend, l_lip):
    """
    Polygon sector nodes.

    Calculates the node coordinates for a cross-section of the shape of
    a lipped polygon sector.

    Parameters
    ----------
    n_sides : int
        Number of sides of original polygon.
    radius : float
        Radius of the original polygon.
    t : float
        Thickness of the profile
    tg : float
        Thickness of the profile
    rbend : float
        Radius of the bended corners' arc
    nbend : int
        Number of nodes along the corners' arcs
    l_lip : int
        Length of the lips

    Returns
    -------
    list of lists
        Returns points for the entire profile (1st and 2nd returned values), and points for a single sector (3rd and 4th
        returned values).

    """

    # Angle corresponding to one face of the polygon
    theta = 2 * np.pi / n_sides

    # Angles of radii (measured from x-axis)
    phi = np.linspace(5 * np.pi / 6, np.pi / 6, int(n_sides / 3 + 1))

    # xy coords of the polygon's corners
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)

    # Bends

    # Distance between bending centre and corner
    lc = rbend / np.cos(theta / 2)

    # Centers of bending arcs
    xc = x[1:-1] - lc * np.cos(phi[1:-1])
    yc = y[1:-1] - lc * np.sin(phi[1:-1])

    # Angles of the edges' midlines (measured from x-axis)
    phi_mids = phi[0:-1] - theta / 2

    # xy coords of the arc's points
    xarc = [[0 for j in range(nbend + 1)] for i in range(int(n_sides / 3 - 1))]
    yarc = [[0 for j in range(nbend + 1)] for i in range(int(n_sides / 3 - 1))]
    for i in range(int(n_sides / 3 - 1)):
        for j in range(nbend + 1):
            xarc[i][j] = xc[i] + rbend * np.cos(phi_mids[i] - j * (theta / nbend))
            yarc[i][j] = yc[i] + rbend * np.sin(phi_mids[i] - j * (theta / nbend))

    # Start-end extensions
    # Bending radius
    rs = rbend / 2
    xcs = [0, 0]
    ycs = [0, 0]

    # First bend
    v1 = phi_mids[0] - np.pi / 2
    v2 = (phi[0] + phi_mids[0] - np.pi / 2) / 2
    l1 = (t + tg) / (2 * np.cos(phi[0] - phi_mids[0]))
    l2 = rs / np.sin(v2 - phi_mids[0] + np.pi / 2)
    x1 = x[0] + l1 * np.cos(v1)
    y1 = y[0] + l1 * np.sin(v1)

    # First bend centre coords
    xcs[0] = x1 + l2 * np.cos(v2)
    ycs[0] = y1 + l2 * np.sin(v2)

    # Last bend
    v1 = phi_mids[-1] + np.pi / 2
    v2 = (v1 + phi[-1]) / 2
    l1 = (t + tg) / (2 * np.cos(v1 - phi[-1] - np.pi / 2))
    l2 = rs / np.sin(v2 - phi[-1])
    x1 = x[-1] + l1 * np.cos(v1)
    y1 = y[-1] + l1 * np.sin(v1)

    # Last bend centre coords
    xcs[1] = x1 + l2 * np.cos(v2)
    ycs[1] = y1 + l2 * np.sin(v2)

    # First and last bend arc points coords
    xsarc = [[0 for j in range(nbend + 1)] for j in [0, 1]]
    ysarc = [[0 for j in range(nbend + 1)] for j in [0, 1]]
    for j in range(nbend + 1):
        xsarc[0][j] = xcs[0] + rs * np.cos(4 * np.pi / 3 + j * ((phi_mids[0] - np.pi / 3) / nbend))
        ysarc[0][j] = ycs[0] + rs * np.sin(4 * np.pi / 3 + j * ((phi_mids[0] - np.pi / 3) / nbend))
        xsarc[1][j] = xcs[1] + rs * np.cos(
            phi_mids[-1] + np.pi + j * ((phi[-1] + np.pi / 2 - phi_mids[-1]) / nbend))
        ysarc[1][j] = ycs[1] + rs * np.sin(
            phi_mids[-1] + np.pi + j * ((phi[-1] + np.pi / 2 - phi_mids[-1]) / nbend))

    # Points of the lips

    # Lip length according to bolt washer diameter

    # First lip
    xstart = [xsarc[0][0] + l_lip * np.cos(phi[0]), xsarc[0][0] + l_lip * np.cos(phi[0]) / 2]
    ystart = [ysarc[0][0] + l_lip * np.sin(phi[0]), ysarc[0][0] + l_lip * np.sin(phi[0]) / 2]

    # Last point
    xend = [xsarc[1][-1] + l_lip * np.cos(phi[-1]) / 2, xsarc[1][-1] + l_lip * np.cos(phi[-1])]
    yend = [ysarc[1][-1] + l_lip * np.sin(phi[-1]) / 2, ysarc[1][-1] + l_lip * np.sin(phi[-1])]

    # Collect the x, y values in a sorted 2xn array
    xarcs, yarcs = [], []
    for i in range(len(phi) - 2):
        xarcs = xarcs + xarc[i][:]
        yarcs = yarcs + yarc[i][:]

    x_sector = xstart + xsarc[0][:] + xarcs[:] + xsarc[1][:] + xend
    y_sector = ystart + ysarc[0][:] + yarcs[:] + ysarc[1][:] + yend

    # Copy-rotate the points of the first sector to create the entire CS
    # Rotation matrix
    rot_matrix = np.array([[np.cos(-2 * np.pi / 3), -np.sin(-2 * np.pi / 3)],
                           [np.sin(-2 * np.pi / 3), np.cos(-2 * np.pi / 3)]])

    # Dot multiply matrices
    coord1 = np.array([x_sector, y_sector])
    coord2 = rot_matrix.dot(coord1)
    coord3 = rot_matrix.dot(coord2)

    # Concatenate into a single xy array
    x_cs = np.concatenate([coord1[0], coord2[0], coord3[0]])
    y_cs = np.concatenate([coord1[1], coord2[1], coord3[1]])

    # Return matrices
    return [x_cs, y_cs, x_sector, y_sector]


def main(
         directory=None,
         add_real_specimens=True,
         add_experimental_data=True,
         make_plots=True,
         export=False,
         print_reports=True
         ):

    if directory is None:
        directory = os.getcwd() + '/data/'

    if export is True:
        export= directory + 'polygonal.pkl'

    # Create a polygonal column object.
    length = 700.
    f_yield = 700.
    fab_class = 'fcA'

    print('Creating the polygonal column objects.')
    cases = [PolygonalColumn(name='specimen{}'.format(i + 1)) for i in range(9)]

    print('Adding theoretical specimens with calculations to the polygonal columns')
    cases[0].add_theoretical_specimen(16, length, f_yield, fab_class, thickness=3., p_class=30.)
    cases[1].add_theoretical_specimen(16, length, f_yield, fab_class, thickness=3., p_class=40.)
    cases[2].add_theoretical_specimen(16, length, f_yield, fab_class, thickness=3., p_class=50.)
    cases[3].add_theoretical_specimen(20, length, f_yield, fab_class, thickness=3., p_class=30.)
    cases[4].add_theoretical_specimen(20, length, f_yield, fab_class, thickness=3., p_class=40.)
    cases[5].add_theoretical_specimen(20, length, f_yield, fab_class, thickness=2., p_class=50.)
    cases[6].add_theoretical_specimen(24, length, f_yield, fab_class, thickness=3., p_class=30.)
    cases[7].add_theoretical_specimen(24, length, f_yield, fab_class, thickness=2., p_class=40.)
    cases[8].add_theoretical_specimen(24, length, f_yield, fab_class, thickness=2., p_class=50.)

    print('Adding real specimens with the 3d scanned data to the polygonal columns.')
    if add_real_specimens:
        for i in range(9):
            print('Adding real scanned shape to specimen number {}'.format(i + 1))
            print(directory + 'sp{}/'.format(i + 1))
            cases[i].add_real_specimen(directory + 'sp{}/'.format(i + 1))

    print('Adding experimental data from the compression tests.')
    if add_experimental_data:
        for i in range(9):
            print('Adding experimental data to specimen number {}'.format(i + 1))
            cases[i].add_experiment(directory + 'sp{}/experiment/sp{}.asc'.format(i + 1, i + 1))

        # Correction of stroke tare value on some measurements.
        cases[1].experiment_data.offset_stroke()
        cases[3].experiment_data.offset_stroke()
        cases[4].experiment_data.offset_stroke()

    if make_plots:
        print('Producing plots.')
        # Strain-stress curves
        ax = cases[0].experiment_data.plot_strain_stress()
        cases[1].experiment_data.plot_strain_stress(ax=ax)
        cases[2].experiment_data.plot_strain_stress(ax=ax)

        ax = cases[3].experiment_data.plot_strain_stress()
        cases[4].experiment_data.plot_strain_stress(ax=ax)
        cases[5].experiment_data.plot_strain_stress(ax=ax)

        ax = cases[6].experiment_data.plot_strain_stress()
        cases[7].experiment_data.plot_strain_stress(ax=ax)
        cases[8].experiment_data.plot_strain_stress(ax=ax)

        # Displacement-load

    if print_reports:
        for i in cases:
            print('')
            i.report_real_specimen()

    if export:
        print('Exporting the generated object with all the processed specimens to pickle.')
        with open(export, 'wb') as fh:
            pickle.dump(cases, fh)

    return cases
