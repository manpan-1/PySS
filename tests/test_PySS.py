#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `PySS` package."""

import unittest

import PySS.polygonal as pg
import PySS.analytic_geometry as ag

from click.testing import CliRunner

from PySS import cli

import os


class TestPySS(unittest.TestCase):
    """Tests for `PySS` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_TheoreticalSpecimen(self):
        """Test the TheoreticalSpecimen object for a polygonal column"""

        # Create a polygonal column object.
        case = pg.PolygonalColumn()

        n_sides = 16
        p_class = 30.
        thickness = 3.
        length = 700.
        f_yield = 700.
        fab_class = 'fcA'

        case.add_theoretical_specimen(n_sides, length, f_yield, fab_class, p_class=p_class, thickness=thickness)

        # Check geometric properties.
        elem = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]
        nodes = [(133.64776466459372,
                  123.47443433950332,
                  94.503240684758083,
                  51.144785309768494,
                  8.1835653604846712e-15,
                  -51.14478530976848,
                  -94.503240684758069,
                  -123.47443433950332,
                  -133.64776466459372,
                  -123.47443433950333,
                  -94.503240684758097,
                  -51.144785309768565,
                  -2.4550696081454014e-14,
                  51.144785309768515,
                  94.503240684758055,
                  123.47443433950329),
                 (0.0,
                  51.144785309768487,
                  94.503240684758069,
                  123.47443433950332,
                  133.64776466459372,
                  123.47443433950332,
                  94.503240684758083,
                  51.144785309768501,
                  1.6367130720969342e-14,
                  -51.144785309768473,
                  -94.503240684758069,
                  -123.47443433950329,
                  -133.64776466459372,
                  -123.4744343395033,
                  -94.503240684758097,
                  -51.144785309768572)]

        self.assertEqual(case.theoretical_specimen.geometry.length, 700.)
        self.assertEqual(case.theoretical_specimen.geometry.r_circle, 132.79066165555551)
        self.assertEqual(case.theoretical_specimen.geometry.thickness, 3.)
        self.assertEqual(case.theoretical_specimen.geometry.cs_sketch.elem, elem)
        self.assertEqual(case.theoretical_specimen.geometry.cs_sketch.nodes, nodes)

        # Check calculations of cross sectional properties.
        self.assertEqual(case.theoretical_specimen.cs_props.area, 2503.0450027345264)
        self.assertEqual(case.theoretical_specimen.cs_props.xc, 0)
        self.assertEqual(case.theoretical_specimen.cs_props.yc, 0)
        self.assertEqual(case.theoretical_specimen.cs_props.min_dist, 131.07976034182852)
        self.assertEqual(case.theoretical_specimen.cs_props.max_dist, 133.64776466459372)
        self.assertEqual(case.theoretical_specimen.cs_props.moi_xx, 21787142.874024708)
        self.assertEqual(case.theoretical_specimen.cs_props.moi_yy, 21787142.874024704)

        # Check calculations of structural properties.
        self.assertEqual(case.theoretical_specimen.struct_props.p_classification, 30.)
        self.assertEqual(case.theoretical_specimen.struct_props.lmbda_y, 0.1378864937380441)
        self.assertEqual(case.theoretical_specimen.struct_props.lmbda_z, 0.13788649373804412)
        self.assertEqual(case.theoretical_specimen.struct_props.n_pl_rd, 1752131.5019141685)
        self.assertEqual(case.theoretical_specimen.struct_props.t_classification, 263.69776782663507)
        self.assertEqual(case.theoretical_specimen.struct_props.n_b_rd_shell, 1426314.6208939506)

    def test_RealSpecimen(self):
        """Test the TheoreticalSpecimen object for a polygonal column"""

        # Create a polygonal column object.
        case = pg.PolygonalColumn()

        n_sides = 16
        p_class = 30.
        thickness = 3.
        length = 700.
        f_yield = 700.
        fab_class = 'fcA'

        cwd = os.getcwd().split(sep='/')[-1]
        if cwd == 'PySS':
            datapath = './test_data/'
        elif cwd == 'tests':
            datapath = './test_data/'
        else:
            datapath = './'

        case.add_theoretical_specimen(n_sides, length, f_yield, fab_class, p_class=p_class, thickness=thickness)
        case.add_real_specimen(datapath)

        # Perform checks
        self.assertEqual(case.real_specimen.thickness, 3.)
        self.assertAlmostEqual(case.real_specimen.sides[0].ref_plane.plane_coeff[0], -0.016774265594874858)
        # self.assertTrue(all(case.real_specimen.sides[0].ref_plane.plane_coeff == [-0.016774265594874858,
        #                                                                           0.99985606971653651,
        #                                                                           0.0025424131751930323,
        #                                                                           130.24190547807578]))

    def test_point_methods(self):
        """Test the `Point3D` class from `analytic_geometry`."""
        a = ag.Line3D.from_2_points([0, 0, 0], [1, 0, 0])
        b = ag.Point3D.from_coordinates(1, 1, 1)
        c = ag.Plane3D.from_coefficients(0, 0, 1, 0)
        self.assertAlmostEqual(b.distance_to_line(a)**2, 2)
        self.assertAlmostEqual(b.distance_to_plane(c), 1.)

    def test_command_line_interface(self):
        """TestData the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'PySS.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output


if __name__ == '__main__':
    unittest.main()
