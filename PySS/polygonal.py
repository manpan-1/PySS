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
import PySS.fem as fem
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#TODO: write extended docstrings for the classes
class PolygonalColumn:
    """
    Polygonal column.

    General class for a polygonal column, including all theoretical, experimental, numerical and 3D scanned data.

    More in detail, "theoretical_specimen" includes geometrical and structural properties and calculations according to
    Eurocodes. The "real_specimen" data holds the 3D scanned object and post processing. Similarly, "experiment_data"
    and "numerical_data" contain the history and processing of laboratory experiments and FEM respectively.

    """

    def __init__(self,
                 name=None,
                 theoretical_specimen=None,
                 real_specimen=None,
                 experiment_data=None,
                 numerical_data=None):
        self.name = name
        self.theoretical_specimen = theoretical_specimen
        self.real_specimen = real_specimen
        self.experiment_data = experiment_data
        self.numerical_data = numerical_data

    def set_theoretical_specimen(self,
                                 n_sides,
                                 length,
                                 f_yield,
                                 fab_class,
                                 r_circle=None,
                                 p_class=None,
                                 thickness=None
                                 ):
        """
        Set a theoretical polygonal column.

        Sets a :obj:`TheoreticalSpecimen` that contains information on the properties and calculations for the polygonal
        column. The calculations are according to Eurocode 3, especially parts 1-1, 1-5 and 1-6.

        The theoretical specimen can be described by its shape or slenderness. In any case, at least 2 of the 3 optional
        input parameters {`r_circle`, `thickness`, `p_class`} must be given.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.
        length : float
            Length of the column.
        f_yield : float
            Yield stress.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN1993-1-6.
        r_circle : float, optional
            Radius of the equicalent cylinder.
        p_class : float, optional
            Plate classification, c/εt.
        thickness : float. optional
            Thickness of the profile.

        """
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
                self.theoretical_specimen = TheoreticalSpecimen.from_pclass_thickness_length(
                    n_sides,
                    p_class,
                    thickness,
                    length,
                    f_yield,
                    fab_class
                )
            else:
                self.theoretical_specimen = TheoreticalSpecimen.from_pclass_radius_length(
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

        Adds a :obj:`RealSpecimen` object from files. Scanned data are loaded from a list of pickle files
        corresponding to sides and edges, each file containing point coordinates. The pickle files are assumed to be in
        the same directory and have the following filename structure:

        - files with points of sides: ``side_XX.pkl``.
        - files with points od edges: ``edge_XX.pkl``

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
        print('Adding centre-line from pickle.')
        specimen.centre_line_from_pickle(path + 'centreline.pkl')

        # Add all sides and edges.
        # they consist of FlatFace and RoundedEdge instances.
        specimen.add_all_facets(n_sides, path + 'side_', fit_planes=True, offset_to_midline=True, calc_lcsys=True)
        # Check if the existing edges found in the directory correspond one by one to the sides. If so, then the
        # intersection lines of adjacent sides are calculated and added to the edges as reference lines. Otherwise, the
        # edges are imported from whatever files are found and no reference lines are calculated.
        if side_filename_numbers == edge_filename_numbers:
            intrsct_lines = True
        else:
            intrsct_lines = False

        specimen.add_all_edges(n_edges, path + 'edge_', intrsct_lines=intrsct_lines)
        # Find a series of points for each edge based on the scanned surface.
        specimen.calc_real_edges(offset_to_midline=True, ref_lines=True)

        # Calculate the initial imperfection displacements based on the edge and facet reference line and plane
        # accordingly.
        specimen.calc_edge_imperfection_displacements()
        specimen.calc_facet_imperfection_displacements()

        # Extract the maximum imperfection displacement from each facet and edge.
        specimen.gather_max_imperfections()

        # Perform fft on edges
        #specimen.fft_all_edges()

        # Assign the constructed specimen to the object
        self.real_specimen = specimen

    def add_experiment(self, fh=None, max_load=None):
        """
        Add and experimental data.

        Adds a :obj:`TestData` object from a file to the polygonal column.

        Parameters
        ----------
        fh : str
            Path to the ascii file containing the experimental data.

        """
        if fh:

            self.experiment_data = TestData.from_file(fh)
            self.experiment_data.specimen_length = self.theoretical_specimen.geometry.length
            self.experiment_data.cs_area = self.theoretical_specimen.cs_props.area
            self.experiment_data.process_data()
        else:
            self.experiment_data = TestData()
        
        if max_load:
            self.max_load = max_load

    def add_numerical(self, filename):
        """
        Add and FEM data.

        Adds a :obj:`fem.FEModel` object from a pickle file containing a history output of a FEM model.

        Parameters
        ----------
        filename : str
            Path to the pickle file.

        """
        self.numerical_data = fem.FEModel.from_hist_pkl(filename)

    # TODO: Fix the report function
    # def report_real_specimen(self):
    #     """Print a report for the processed scanned data of the real specimen."""
    #     print('Report for {}'.format(self.name))
    #     self.real_specimen.print_report()


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
            r_cyl,
            thickness,
            length,
            f_y,
            fab_class,
            a_b=3.
    ):
        """
        Create theoretical polygonal column object for given geometric data.

        The constructor calculates properties of the polygonal column object (cross-section props,
        resistance, geometric props etc). The calculated data is then used to construct an object.

        This is the basic alternative constructor, several other alternative constructors are defined for different
        cases of imput data. All of the following alternative constructors are calling this one to create the object
        after having performed the necessary geometrical pre-calculations.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.
        r_cyl : float
            Radius of the circle circumscribed to the polygon.
        thickness : float
            Thickness of the cross-section.
        length : float
            Length of the column.
        f_y : float
            Yield stress of the material.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.
        a_b : float
            Corner bending radius over thickness ratio

        """

        # Create material
        material = sd.Material(210000., 0.3, f_y)

        # Bending radius
        r_b = a_b * thickness
        
        # Theta angle
        theta = np.pi / n_sides

        # Radius of the polygon's circumscribed circle
        r_p = (np.pi * r_cyl + a_b * thickness * (n_sides * np.tan(theta) - np.pi)) / (n_sides * np.sin(theta))

        # Width of each side
        bbbb = 2 * r_p * np.sin(theta)

        # Width of the corner bend half arc projection on the plane of the facet
        b_c = r_b * np.tan(theta)

        # Flat width of each facet (excluding the bended arcs)
        cccc = bbbb - 2 * b_c
        
        # Cross sectional area
        area = 2 * np.pi * r_cyl * thickness
        
        # Moment of inertia
        b_o = bbbb + thickness * np.tan(theta)
        alfa = thickness * np.tan(theta) / b_o
        moi = (n_sides * b_o ** 3 * thickness / 8) * (1 / 3 + 1 / (np.tan(theta) ** 2)) * (1 - 3 * alfa + 4 * alfa ** 2 - 2 * alfa ** 3)
        
        # Effective vross secion area
        corner_area = 2 * np.pi * r_b * thickness
        a_eff = n_sides * sd.calc_a_eff(thickness, cccc, f_y) + corner_area
        
        # Gather all cross sectional properties in an appropriate class
        cs_props = sd.CsProps(
            area=area,
            a_eff=a_eff,
            xc=0.,
            yc=0.,
            moi_xx=moi,
            moi_yy=moi,
            moi_xy=0.,
            moi_1=moi,
            moi_2=moi)

        # cs_props = sd.CsProps.from_cs_sketch(cs_sketch)
        cs_props.max_dist = r_p
        cs_props.min_dist = np.sqrt(r_p ** 2 - (bbbb / 2) ** 2)
        
        # Polar coordinate of the polygon vertices on the cross-section plane
        phii = []
        for i_index in range(n_sides):
            phii.append(i_index * theta)

        # Polygon corners coordinates.
        x_corners = tuple(r_p * np.cos(phii))
        y_corners = tuple(r_p * np.sin(phii))

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
        geometry.r_circle = r_cyl
        geometry.r_circumscribed = r_p
        geometry.facet_width = bbbb
        geometry.facet_flat_width = cccc
        geometry.n_sides = n_sides
        geometry.r_bend = r_b

        lmbda_y = sd.lmbda_flex(
            length,
            cs_props.a_eff,
            cs_props.moi_1,
            kapa_bc=1.,
            e_modulus=material.e_modulus,
            f_yield=material.f_yield
        )

        lmbda_z = lmbda_y

        # Plate classification (acc. to EC3-1-1)
        p_classification = cccc / (material.epsilon * thickness)

        # Critical stress acc. to plate theory.
        sigma_cr_plate = sd.sigma_cr_plate(thickness, bbbb - 3*thickness)

        # Critical load acc. to plate theory.
        n_cr_plate = cs_props.area * sigma_cr_plate

        # Axial compression resistance, Npl (acc. to EC3-1-5)
        n_pl_rd = cs_props.a_eff * f_y

        # Buckling load
        n_b_rd = sd.n_b_rd(geometry.length, cs_props.a_eff, cs_props.moi_1, f_y, "d")
        
        # Buckling stress (account for both flex and local)
        sigma_b_rd_plate = n_b_rd / cs_props.area

        # Tube classification slenderness acc. to EC3-1-1
        t_classification = 2 * r_cyl / (material.epsilon ** 2 * thickness)

        # Length categorisation acc. to EC3-1-1, new draft proposal
        lenca = sd.shell_length_category(r_cyl, thickness, length)
        lenca_new = sd.shell_length_category_new(r_cyl, thickness, length)
        
        # Critical stress acc. to shell theory.
        sigma_cr_shell = sd.sigma_x_rcr(thickness, r_cyl, length)
        sigma_cr_shell_new = sd.sigma_x_rcr_new(thickness, r_cyl, length)
        
        # Critical load acc. to shell theory.
        n_cr_shell = sd.n_cr_shell(thickness, r_cyl, length)
        n_cr_shell_new = sd.n_cr_shell_new(thickness, r_cyl, length)

        # Compression stress of equivalent cylindrical shell (acc. to EC3-1-6)
        sigma_b_rd_shell = sd.sigma_x_rd(thickness, r_cyl, length, f_y, fab_quality=fab_class)
        sigma_b_rd_shell_new = sd.sigma_x_rd_new(thickness, r_cyl, length, f_y, fab_quality=fab_class)
        
        # Compression resistance of equivalent cylindrical shell (acc. to EC3-1-6)
        n_b_rd_shell = cs_props.area * sigma_b_rd_shell
        n_b_rd_shell_new = cs_props.area * sigma_b_rd_shell_new
        
        struct_props = sd.StructProps(
            t_classification=t_classification,
            p_classification=p_classification,
            lmbda_y=lmbda_y,
            lmbda_z=lmbda_z,
            n_cr_plate=n_cr_plate,
            sigma_cr_plate=sigma_cr_plate,
            n_pl_rd=n_pl_rd,
            sigma_b_rd_plate=sigma_b_rd_plate,
            n_b_rd_plate=n_b_rd,
            sigma_cr_shell=sigma_cr_shell,
            sigma_cr_shell_new=sigma_cr_shell_new,
            lenca=lenca,
            lenca_new=lenca_new,
            n_cr_shell=n_cr_shell,
            n_cr_shell_new=n_cr_shell_new,
            sigma_b_rd_shell=sigma_b_rd_shell,
            sigma_b_rd_shell_new=sigma_b_rd_shell_new,
            n_b_rd_shell=n_b_rd_shell,
            n_b_rd_shell_new=n_b_rd_shell_new
        )


        return cls(geometry, cs_props, material, struct_props)

    @classmethod
    def from_pclass_thickness_length(
            cls,
            n_sides,
            p_classification,
            thickness,
            length,
            f_y,
            fab_class,
            a_b=3.
    ):
        """
        Create theoretical polygonal column object for given plate slenderness, thickness and length.

        Uses the :func:`~PySS.polygonal.from_geometry`.

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
        f_y : float
            Yield stress of the material.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.
        a_b : float
            Thickness to bending radius ratio.

        """

        # Epsilon for the material
        epsilon = np.sqrt(235. / f_y)

        # Radius of the equal perimeter cylinder
        #r_circle = (n_sides * thickness / np.pi) * ((p_classification * epsilon / 2) + arc_to_thickness * np.tan(np.pi / n_sides))
        r_circle = thickness*(n_sides * p_classification * epsilon / (2*np.pi) + a_b)

        return cls.from_geometry(
            n_sides,
            r_circle,
            thickness,
            length,
            f_y,
            fab_class
        )

    @classmethod
    def from_pclass_radius_length(
            cls,
            n_sides,
            r_cyl,
            p_classification,
            length,
            f_y,
            fab_class,
            a_b=3.
    ):
        """
        Create theoretical polygonal column object for given equivalent cylinder radius, plate slenderness and length.

        Uses the :func:`~PySS.polygonal.from_geometry`.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.
        r_cyl : float
            Radius of the equivalent cylinder.
        p_classification : float
            Facet slenderness, c/(ε*t).
        length : float
            Length of the column.
        f_y : float
            Yield stress of the material.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.
        a_b : float
            Thickness to bending radius ratio.

        """

        # Epsilon for the material
        epsilon = np.sqrt(235. / f_y)

        # Calculate the thickness
        thickness = r_cyl / ((n_sides * p_classification * epsilon / (2 * np.pi)) + a_b)

        return cls.from_geometry(
            n_sides,
            r_cyl,
            thickness,
            length,
            f_y,
            fab_class
        )
    
    @classmethod
    def from_pclass_area_length(
            cls,
            n_sides,
            p_classification,
            area,
            length,
            f_y,
            fab_class,
            a_b=3
    ):
        """
        Create theoretical polygonal column object for given equivalent cylinder radius, plate slenderness and
        length.

        Uses the :func:`~PySS.polygonal.from_geometry`.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.
        p_classification : float
            Facet slenderness, c/(ε*t).
        area : float
            Cross-sectional area.
        length : float
            Length of the column.
        f_y : float
            Yield stress of the material.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.
        a_b : float
            Thickness to bending radius ratio.

        """

        # Epsilon for the material
        epsilon = np.sqrt(235. / f_y)
        
        # Thickness
        thickness = np.sqrt(area / (n_sides * p_classification * epsilon + 2 * np.pi * a_b))
        
        # Radius of equivalent cylinder
        r_cyl  = area / (2 * np.pi * thickness)
        
        return cls.from_geometry(
            n_sides,
            r_cyl,
            thickness,
            length,
            f_y,
            fab_class
        )

    @classmethod
    def from_radius_area_length(
            cls,
            n_sides,
            r_cyl,
            area,
            length,
            f_y,
            fab_class,
            a_b=3
    ):
        """
        Create theoretical polygonal column object for given equivalent cylinder radius, area and length.

        Uses the :func:`~PySS.polygonal.from_geometry`.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.
        r_cyl : float
            Radius of the equivalent cylinder.
        area : float
            Cross-sectional area.
        length : float
            Length of the column.
        f_y : float
            Yield stress of the material.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.
        a_b : float
            Thickness to bending radius ratio.

        """

        thickness = area / (2 * np.pi *r_cyl)

        return cls.from_geometry(
            n_sides,
            r_cyl,
            thickness,
            length,
            f_y,
            fab_class,
            a_b=a_b
        )

    @classmethod
    def from_radius_thickness_flexslend(
            cls,
            n_sides,
            r_cyl,
            thickness,
            lambda_flex,
            f_y,
            fab_class,
            a_b=3.
    ):
        """
        Create theoretical polygonal column object for given equivalent cylinder radius, thickness and flexural
        slenderness.

        Uses the :func:`~PySS.polygonal.from_geometry`.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.
        r_cyl : float
            Radius of the equivalent cylinder.
        thickness : float
            Thickness of the cross-section.
        lambda_flex : float
            Flexural slenderness.
        f_y : float
            Yield stress of the material.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.
        a_b : float
            Thickness to bending radius ratio.

        """

        # Bending radius
        r_b = a_b * thickness

        # Theta angle
        theta = np.pi / n_sides

        # Radius of the polygon's circumscribed circle
        r_p = (np.pi * r_cyl + a_b * thickness * (n_sides * np.tan(theta) - np.pi)) / (n_sides * np.sin(theta))

        # Width of each side
        bbbb = 2 * r_p * np.sin(theta)

        # Width of the corner bend half arc projection on the plane of the facet
        b_c = r_b * np.tan(theta)

        # Flat width of each facet (excluding the bended arcs)
        cccc = bbbb - 2 * b_c

        # Moment of inertia
        b_o = bbbb + thickness * np.tan(theta)
        alfa = thickness * np.tan(theta) / b_o
        moi = (n_sides * b_o ** 3 * thickness / 8) * (1 / 3. + 1 / (np.tan(theta) ** 2)) * (
                    1 - 3 * alfa + 4 * alfa ** 2 - 2 * alfa ** 3)
        # Effective cross secion area
        corner_area = 2 * np.pi * r_b * thickness
        a_eff = n_sides * sd.calc_a_eff(thickness, cccc, f_y) + corner_area

        # Calculate column length for the given flexural slenderness.
        length = lambda_flex * np.pi * np.sqrt(210000. * moi / (a_eff * f_y))

        return cls.from_geometry(
            n_sides,
            r_cyl,
            thickness,
            length,
            f_y,
            fab_class,
            a_b=a_b
        )
    
    @classmethod
    def from_pclass_radius_flexslend(
            cls,
            n_sides,
            r_cyl,
            p_classification,
            lambda_flex,
            f_y,
            fab_class,
            a_b=3.
    ):
        """
        Create theoretical polygonal column object for given equivalent cylinder radius, plate classification and
        flexural slenderness.

        Uses the :func:`~PySS.polygonal.from_geometry`.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.
        r_cyl : float
            Radius of the equivalent cylinder.
        p_classification : float
            Facet slenderness, c/(ε*t).
        lambda_flex : float
            Flexural slenderness.
        f_y : float
            Yield stress of the material.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.
        a_b : float
            Thickness to bending radius ratio.

        """

        # Epsilon for the material
        epsilon = np.sqrt(235. / f_y)

        # Calculate the thickness
        thickness = r_cyl / ((n_sides * p_classification * epsilon / (2 * np.pi)) + a_b)

        return cls.from_radius_thickness_flexslend(
            n_sides,
            r_cyl,
            thickness,
            lambda_flex,
            f_y,
            fab_class,
            a_b=a_b
        )
    
    @classmethod
    def from_pclass_area_flexslend(
            cls,
            n_sides,
            p_classification,
            area,
            lambda_flex,
            f_y,
            fab_class,
            a_b=3.
    ):
        """
        Create theoretical polygonal column object for given plate classification, area and flexural slenderness.

        Uses the :func:`~PySS.polygonal.from_geometry`.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon cross-section.
        p_classification : float
            Facet slenderness, c/(ε*t).
        area : float
            Cross-sectional area.
        lambda_flex : float
            Flexural slenderness.
        f_y : float
            Yield stress of the material.
        fab_class : {'fcA', 'fcB', 'fcC'}
            Fabrication class, as described in EN 1996-1-6. It is used in the calculation of the buckling resistance of
            the cylinder of equal thickness-perimeter.
        a_b : float
            Thickness to bending radius ratio.

        """
        # Epsilon for the material
        epsilon = np.sqrt(235. / f_y)

        # Thickness
        thickness = np.sqrt(area / (n_sides * p_classification * epsilon + 2 * np.pi * a_b))

        # Radius of equivalent cylinder
        r_cyl = area / (2 * np.pi * thickness)
        
        return cls.from_radius_thickness_flexslend(
            n_sides,
            r_cyl,
            thickness,
            lambda_flex,
            f_y,
            fab_class,
            a_b=3
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
        self.u_edges = None

    def centre_line_from_pickle(self, fh):
        """
        Import a centre-line to the polygonal object from a pickle file.

        The pickle file is expected to contain a list of 2 points from which the line is constructed. This method is
        used in combination with the equivalent ``export`` method from blender.

        Parameters
        ----------
        fh : str
            Path and filename of the pickle file.

        """
        self.centre_line = ag.Line3D.from_pickle(fh)

    def add_single_facet_from_pickle(self, filename):
        """
        Create a FlatFace instance as one side af the polygon column.

        The FlatFace instance is created from a pickle file of scanned data points.

        Parameters
        ----------
        filename : str
            Path and filename of the pickle file.

        """
        self.sides.append(s3d.FlatFace.from_pickle(filename))

    def add_all_facets(self, n_sides, prefix, fit_planes=False, offset_to_midline=False, calc_lcsys=False):
        """
        Add multiple sides.

        Multiple FlatFace instances are created as sides of the polygonal column. A series of files containing scanned
        data points must be given. The files should be on the same path and have a filename structure as:
        ``path/basenameXX.pkl``, where ``XX`` is an id number in ascending order starting from 01.
        Only the ``path/basename`` is given as input to this method.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygonal cross-section to look for in the directory
        prefix : str
            Path and file name prefix for the pickle files containing the scanned data points.
        fit_planes :
            Perform least square fitting on the imported data to calculate the reference planes.
        offset_to_midline : bool
            Offset the data points and the fitted plane by half the thickness to be on the midline of the cross-section.
        calc_lcsys : bool
            Translate points to the local csys. They are stored separetely in "points_lcsys"

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

        if calc_lcsys:
            for i, x in enumerate(self.sides):
                print('Transform points to local csys, facet:    {}'.format(i + 1))
                x.local_from_world()

    def add_single_edge_from_pickle(self, filename):
        """
        Create a RoundEdge instance as one edges af the polygon column.

        The RoundEdge instance is created from a pickle file of scanned data points.

        Parameters
        ----------
        filename : str
            Path and filename of the pickle file.

        """
        self.edges.append(s3d.RoundedEdge.from_pickle(filename))

    def add_all_edges(self, n_sides, prefix, intrsct_lines=False):
        """
        Add multiple edges.

        Multiple :obj:`scan_3D.RoundedEdge` objects are created as edges of the polygonal column. A series of files
        containing scanned data points must be given. The files should be on the same path and have a filename structure
        as:
        ``path/basenameXX.pkl``, where ``XX``is an id number in ascending order starting from 01.
        Only the 'path/basename' should be given for the 'prefix' input argument. The ``XX.pkl`` part is set
        automatically .

        After adding the sequential edges, if ``intrsct_lines`` is ``True``, the reference lines are calculated as the
        intersections of sequential facets of the polygonal specimen.

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

    def calc_real_edges(self, offset_to_midline=False, ref_lines=False):
        """
        Calculate edge points on the scanned rounded edge.

        Batches of points are calculated which represent the real edges of the polygonal column. Each point is
        calculated as  the intersection of a circle and a line at different heights of the column, where the circle is
        best fit to the rounded edge scanned points and the line passing through the theoretical edge and the polygon's
        centre line.

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

    def calc_edge_imperfection_displacements(self):
        """
        Calculate the initial imperfection displacements of the edges.

        The initial imperfection displacements are calculated as the distance of all the real edge points to the
        their respective reference lines. The distances are signed, being positive for points that are further away
        from the centre-line than the reference line at the same height.

        Notes
        -----
        For the method to function, all edges must contain ``real_edge`` and ``ref_lines`` attributes.

        See Also
        --------
        calc_real_edges : Method providing ``real_edge`` and ``ref_line``.
        calc_facet_imperfection_displacements : Equivalent method for the facet imperfections.

        """
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

    #TODO: Signed imperfections and plotting.
    def calc_facet_imperfection_displacements(self):
        """
        Calculate the initial imperfection displacements of the facets.

        Initial imperfection displacements are calculated as the distance of all the scanned data points to their
        respective reference plane.

        Notes
        -----
        For the method to function, all facets need to have ``swarm`` and ``ref_plane`` attributes.

        See Also
        --------
        add_all_sides : Method providing ``swarm`` and ``ref_plane``.
        calc_edge_imperfection_displacements : Equivalent method for the edge imperfections.

        """
        for i, x in enumerate(self.sides):
            print('Calculating initial imperfection displacements, facet:    {}'.format(i + 1))
            x.calc_face2ref_dist()

    # def fft_all_edges(self):
    #     edges_u_max = []
    #     for i in self.edges:
    #         i.fft()
    #         edges_u_max.append(i.u_max)
    #
    #     self.u_edges = edges_u_max

    def plot_all(self):
        """
        Plot all data.

        Plots all the facets, both scanned data points and fitted reference planes, and the theoretical edges (plane
        intersections).

        """
        fig1 = plt.figure()
        Axes3D(fig1)
        for i in range(-len(self.sides), 0):
            self.sides[i].scatter_face(reduced=0.01, fig=fig1)
        for i in self.edges:
            max_z = max([i.coords[2] for i in i.points_wcsys.swarm])
            min_z = min([i.coords[2] for i in i.points_wcsys.swarm])

            i.theoretical_edge.plot_line(fig=fig1, ends=[min_z, max_z])

    def gather_max_imperfections(self):
        """
        Collect initial imperfection info from all the edges and facets.

        Calculates the maximum imperfection of all facets and edges saves the lists at the equivalent class attributes.

        """
        self.max_face_imp = []
        self.max_edge_imp = []
        for x in self.sides:
            self.max_face_imp.append(max(np.abs(x.face2ref_dist)))
        for x in self.edges:
            try:
                self.max_edge_imp.append(max(np.abs(x.edge2ref_dist[1])))
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
        # max_z = max([x.swarm[:, 2].max() for x in self.sides])
        # min_z = min([x.swarm[:, 2].min() for x in self.sides])
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

    def plot_facets(self):
        """
        Plot all the facets together
        """
        for i in self.sides:
            i.regularise_grid()

        fig = plt.figure()
        for i, facet in enumerate(self.sides):
            print(i)
            fig.add_subplot(16, 1, i + 1)
            plt.imshow(facet.regular_grid[2], cmap='gray')
            plt.title(str(i)), plt.xticks([]), plt.yticks([])


class TestData(lt.Experiment):
    """
    Laboratory test of polygonal specimen.

    """
    def __init__(self, name=None, channels=None, specimen_length=None, cs_area=None, max_load=None):
        self.specimen_length = specimen_length
        self.cs_area = cs_area
        self.max_load = max_load

        super().__init__(name=name, channels=channels)
        
        if channels:
            facets = [int(i[:2]) for i in self.channels.keys() if len(i) == 3 and i[2] == "F"]
            edges = [int(i[:2]) for i in self.channels.keys() if len(i) == 3 and i[2] == "C"]
            self.facets_with_gauges = facets
            if len(edges)<len(facets):
                self.facets_with_gauges = edges
        else:
            self.facets_with_gauges = None

    def process_data(self):
        """
        Process raw data.

        Method simply calling all the relevant processing methods, which are:
        1. :obj:`calc_avg_strain`
        2. :obj:`calc_disp_from_strain`
        3. :obj:`calc_avg_stress`

        """
        self.calc_avg_strain()
        self.calc_disp()
        self.calc_avg_stress()

    #TODO: fix the add_Eccentricity method
    # def add_eccentricity(self, axis, column, moi, min_dist, thickness, young):
    #     """
    #     Calculate eccentricity.
    #
    #     Adds a column in the data dictionary for the eccentricity of the load application on a given axis based on
    #     two opposite strain measurements.
    #
    #     Parameters
    #     ----------
    #     axis :
    #     column :
    #     moi :
    #     mon_dist :
    #     thickness :
    #     young :
    #
    #     """
    #
    #     self.channels['e_' + axis] = {}
    #     self.channels['e_' + axis]["data"] = []
    #     self.channels['e_' + axis]["units"] = "mm"
    #     for load, strain1, strain2 in zip(self.channels['Load']["data"],
    #                                       self.channels[column[0]]["data"],
    #                                       self.channels[column[1]]["data"]):
    #         self.channels['e_' + axis]["data"].append(self.eccentricity_from_strain(
    #             load * 1000,
    #             [strain1 * 1e-6, strain2 * 1e-6],
    #             moi,
    #             min_dist + thickness / 2,
    #             young)
    #         )

    def offset_stroke(self, offset=None):
        """
        Offset stroke values.

        Used to set the tare value of the stroke. The required offset can be either given use the first reading of the
        stroke channel.

        Parameters
        ----------
        offset : float, optional
            Distance to offset. By default, the initial position (first value) is used, effectively displacing
            the values to start from 0.

        """
        if offset is None:
            offset = self.channels['Stroke']["data"][0]

        self.channels['Stroke']["data"] = self.channels['Stroke']["data"] - offset

    def calc_disp(self):
        """Calculate the specimen clear axial deformation based on measured strains and on the LVDTs"""
        self.add_new_channel_zeros('disp_from_strain', "mm")
        self.channels['disp_from_strain']["data"] = self.channels['avg_strain']["data"] * self.specimen_length

        self.add_new_channel_zeros('disp_from_lvdt', "mm")
        for i in range(4):
            self.channels["disp_from_lvdt"]["data"] = self.channels["disp_from_lvdt"]["data"] + \
                                                 self.channels["LVDT{}".format(i + 1)]["data"]
        self.channels["disp_from_lvdt"]["data"] = self.channels["disp_from_lvdt"]["data"] / 4.
        
    def calc_avg_strain(self):
        """Calculate the average strain from all strain gauges."""
        # Create new data channel.
        self.add_new_channel_zeros('avg_strain', "mm/mm")
        i = 0
        # Collect all strain gauge records.
        for key in self.channels.keys():
            if len(key) > 2:
                if key[:2].isdigit() and (key[2] is 'F') or (key[2] is 'C'):
                    self.channels['avg_strain']["data"] = self.channels['avg_strain']["data"] + self.channels[key]["data"]
                    i += 1

        self.channels['avg_strain']["data"] = self.channels['avg_strain']["data"] / (i * 1e6)
        
    def plot_load_strain_gauges(self):
        """Plot all the stain gauge channels against the load"""

        fig1 = plt.figure()
        plt.plot()
        ax1 = fig1.axes[0]
        fig2 = plt.figure()
        plt.plot()
        ax2 = fig2.axes[0]
        
        # Collect all strain gauge records.
        for key in self.channels.keys():
            if len(key) > 2:
                if key[:2].isdigit() and (key[2] is 'F'):
                    self.plot2d(key, "Load", scale=(-1, -1), ax=ax1)
                elif key[:2].isdigit() and (key[2] is 'C'):
                    self.plot2d(key, "Load", scale=(-1, -1), ax=ax2)
                    
    def plot_strain_gage_pair(self, facet_nr, ax=None):
        """Plot a pair of face-edge strain gauges against the load"""
        
        if ax is None:
            fig = plt.figure()
            plt.plot()
            ax = fig.axes[0]
            self.plot2d("{:02d}C".format(facet_nr),"Load", scale=(-1, -1), ax=ax)
            self.plot2d("{:02d}F".format(facet_nr),"Load", scale=(-1, -1), ax=ax)
            ax.grid()
            return ax
        elif not isinstance(ax, type(plt.axes())):
            print('Unexpected input type. Input argument `ax` must be of type `matplotlib.pyplot.axes()`')
            return NotImplemented
        else:
            self.plot2d("{:02d}C".format(facet_nr),"Load", scale=(-1, -1), ax=ax)
            self.plot2d("{:02d}F".format(facet_nr),"Load", scale=(-1, -1), ax=ax)
            return ax

    def calc_avg_stress(self):
        """Calculate the average stress from the measured reaction force and the cross-section area."""
        # Create new data channel.
        self.add_new_channel_zeros('avg_stress', "Mpa")
        self.channels['avg_stress']["data"] = self.channels['Load']["data"] * 1e3 / self.cs_area

    def plot_stroke_load(self, ax=None):
        """
        Load vs stroke curve plotter

        Parameters
        ----------
        ax : :obj:`matplotlib.axes`, optional
            Axes for the plot to be added in. By default is plotted on a new figure.

        """
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
        """
        Plot average strain vs average stress.

        Parameters
        ----------
        ax : :obj:`matplotlib.axes`, optional
            Axes for the plot to be added in. By default is plotted on a new figure.

        """
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
        """
        Plot load vs real displacement.

        Parameters
        ----------
        ax : :obj:`matplotlib.axes`, optional
            Axes for the plot to be added in. By default is plotted on a new figure.

        """
        if ax is None:
            fig = plt.figure()
            plt.plot()
            ax = fig.axes[0]
            self.plot2d('disp_from_lvdt', 'Load', ax=ax)
            self.plot2d("disp_from_strain", "Load", ax=ax)
            ax.invert_xaxis()
            ax.invert_yaxis()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(["LVDT average", "strain * length average"])
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

    #TODO: fix the eccentricity method.
    # @staticmethod
    # def eccentricity_from_strain(load, strain, moi, dist, young=None):
    #     """
    #     Load eccentricity based on strain pairs.
    #
    #     Calculate the eccentricity of an axial load to the neutral axis of a specimen for which pairs of strains are
    #     monitored with strain gauges. The eccentricity is calculated on one axis and requires the moment of inertia
    #     around it and a pair of strains on tow positions symmetric to the neutral axis. Elastic behaviour is assumed.
    #
    #     """
    #
    #     # Default values.
    #     if young is None:
    #         young = 210000.
    #     else:
    #         young = float(young)
    #
    #     # Eccentricity.
    #     ecc = (strain[0] - strain[1]) * young * moi / (2 * load * dist)
    #
    #     # Return
    #     return ecc


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
         nominal=True,
         add_real_specimens=True,
         add_experimental_data=True,
         add_numerical_data=True,
         make_plots=False,
         export=False,
         ):

    if directory is None:
        directory = os.getcwd() + '/data/'

    if export is True:
        export= directory + 'polygonal.pkl'

    # Create a polygonal column object.
    if nominal:
        print('Using nominal values for thickness and yield stress.')
        f_yield = 700.
        thickness1 = 3.
        thickness2 = 2.
    else:
        print('Using measured values for thickness and yield stress.')
        #TODO: Calculate the field stress accurately from the coupons using the 0.002 rule.
        f_yield = 715.
        thickness1 = 3.0385
        thickness2 = 1.886

    length = 700.
    fab_class = 'fcA'

    print('Creating the polygonal column objects.')
    cases = [PolygonalColumn(name='specimen{}'.format(i + 1)) for i in range(9)]

    print('Adding theoretical specimens with calculations to the polygonal columns')
    cases[0].set_theoretical_specimen(16, length, f_yield, fab_class, thickness=thickness1, p_class=30.)
    cases[1].set_theoretical_specimen(16, length, f_yield, fab_class, thickness=thickness1, p_class=40.)
    cases[2].set_theoretical_specimen(16, length, f_yield, fab_class, thickness=thickness1, p_class=50.)
    cases[3].set_theoretical_specimen(20, length, f_yield, fab_class, thickness=thickness1, p_class=30.)
    cases[4].set_theoretical_specimen(20, length, f_yield, fab_class, thickness=thickness1, p_class=40.)
    cases[5].set_theoretical_specimen(20, length, f_yield, fab_class, thickness=thickness2, p_class=50.)
    cases[6].set_theoretical_specimen(24, length, f_yield, fab_class, thickness=thickness1, p_class=30.)
    cases[7].set_theoretical_specimen(24, length, f_yield, fab_class, thickness=thickness2, p_class=40.)
    cases[8].set_theoretical_specimen(24, length, f_yield, fab_class, thickness=thickness2, p_class=50.)

    if add_real_specimens:
        print('Adding real specimens with the 3d scanned data to the polygonal columns.')
        for i in range(9):
            print('Adding real scanned shape to specimen number {}'.format(i + 1))
            cases[i].add_real_specimen(directory + 'sp{}/'.format(i + 1))

    if add_experimental_data:
        print('Adding experimental data from the compression tests.')
        for i in range(9):
            print('Adding experimental data to specimen number {}'.format(i + 1))
            cases[i].add_experiment(fh=directory + 'sp{}/experiment/sp{}.asc'.format(i + 1, i + 1))
            cases[i].experiment_data.max_load = -cases[i].experiment_data.channels["Load"]["data"].min() * 1000

        # Correction of stroke tare value on some measurements.
        cases[1].experiment_data.offset_stroke()
        cases[3].experiment_data.offset_stroke()
        cases[4].experiment_data.offset_stroke()

    if add_numerical_data:
        print('Adding data from numerical analyses.')
        for i in range(9):
            print('Adding numerical data to specimen number {}'.format(i + 1))
            print(directory + 'sp{}/'.format(i + 1))
            cases[i].add_numerical(directory + 'fem/sp{}'.format(i + 1))

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

    # TODO: fix the reporting functions
    # if print_reports:
    #     for i in cases:
    #         print('')
    #         i.report_real_specimen()

    if export:
        print('Exporting the generated object with all the processed specimens to pickle.')
        with open(export, 'wb') as fh:
            pickle.dump(cases, fh)

    return cases
