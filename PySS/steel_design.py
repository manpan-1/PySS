# -*- coding: utf-8 -*-

"""
Module for the structural design of steel members.

"""

import numpy as np


class Geometry:
    """
    Structural element geometry.

    Class for the geometric properties of a structural element.

    Parameters
    ----------
    cs_sketch : CsSketch object
        Cross-section sketch.
    length : float
        Member's length.

    """

    def __init__(self, cs_sketch, length, thickness):
        self.cs_sketch = cs_sketch
        self.length = length
        self.thickness = thickness


class CsSketch:
    """
    Cross-section geometry.

    Parameters
    ----------
    nodes : list
        List of points.
    elem : list
        Element connectivity.

    """

    def __init__(self, nodes, elem):
        self.nodes = nodes
        self.elem = elem


class CsProps:
    """
    Cross-section properties.

    Class for the mass properties of cross-sections. The properties can be calculated using the from_cs_sketch() method.

    Parameters
    ----------
    area : float
        Cross-sectional area.
    xc : float
        `x` coordinate of the gravity center.
    yc : float
        `y` coordinate of the gravity center.
    moi_xx : float
        Moment of inertia around `x` axis.
    moi_yy : float
        Moment of inertia around `y` axis.
    moi_xy : float
        Polar moment of inertia.
    theta_principal : float
        Rotation of the principal axes.
    moi_1 : float
        Moment of inertia around the major axis.
    moi_2 : float
        Moment of inertia around the minor axis.

    """

    def __init__(self,
                 area=None,
                 a_eff=None,
                 xc=None,
                 yc=None,
                 moi_xx=None,
                 moi_yy=None,
                 moi_xy=None,
                 theta_principal=None,
                 moi_1=None,
                 moi_2=None
                 ):

        self.area = area
        self.a_eff = a_eff
        self.xc = xc
        self.yc = yc
        self.moi_xx = moi_xx
        self.moi_yy = moi_yy
        self.moi_xy = moi_xy
        self.theta_principal = theta_principal
        self.moi_1 = moi_1
        self.moi_2 = moi_2

    @classmethod
    def from_cs_sketch(cls, cs_sketch):
        """
        Cross-section calculator.

        Alternative constructor, calculates mass properties of a given sc sketch and returns a CsProps object.

        Parameters
        ----------
        cs_sketch : CsSketch object

        Notes
        -----

        """

        nele = len(cs_sketch.elem[0])
        node = cs_sketch.elem[0] + cs_sketch.elem[1]
        nnode = 0
        j = 0

        while node:
            i = [ii for ii, x in enumerate(node) if x == node[0]]
            for ii in sorted(i, reverse=True):
                del node[ii]
            if len(i) == 2:
                j += 1
            nnode += 1

        # classify the section type (currently not used)
        # if j == nele:
        #     section = 'close'  # single cell
        # elif j == nele - 1:
        #     section = 'open'  # singly-branched
        # else:
        #     section = 'open'  # multi-branched

        # Calculate the cs-properties
        tt = []
        xm = []
        ym = []
        xd = []
        yd = []
        side_length = []
        for i in range(nele):
            sn = cs_sketch.elem[0][i]
            fn = cs_sketch.elem[1][i]
            # thickness of the element
            tt = tt + [cs_sketch.elem[2][i]]
            # compute the coordinate of the mid point of the element
            xm = xm + [mean_list([cs_sketch.nodes[0][sn], cs_sketch.nodes[0][fn]])]
            ym = ym + [mean_list([cs_sketch.nodes[1][sn], cs_sketch.nodes[1][fn]])]
            # compute the dimension of the element
            xd = xd + [(cs_sketch.nodes[0][fn] - cs_sketch.nodes[0][sn])]
            yd = yd + [(cs_sketch.nodes[1][fn] - cs_sketch.nodes[1][sn])]
            # compute the length of the element
            side_length = side_length + [np.sqrt(xd[i] ** 2 + yd[i] ** 2)]

        # calculate cross sectional area
        area = sum([a * b for a, b in zip(side_length, tt)])
        # compute the centroid
        xc = sum([a * b * c for a, b, c in zip(side_length, tt, xm)]) / area
        yc = sum([a * b * c for a, b, c in zip(side_length, tt, ym)]) / area

        if abs(xc / np.sqrt(area)) < 1e-12:
            xc = 0

        if abs(yc / np.sqrt(area)) < 1e-12:
            yc = 0

        # Calculate MOI
        moi_xx = sum([sum(a) for a in zip([a ** 2 * b * c / 12 for a, b, c in zip(yd, side_length, tt)],
                                          [(a - yc) ** 2 * b * c for a, b, c in
                                           zip(ym, side_length, tt)])])
        moi_yy = sum([sum(a) for a in zip([a ** 2 * b * c / 12 for a, b, c in zip(xd, side_length, tt)],
                                          [(a - xc) ** 2 * b * c for a, b, c in
                                           zip(xm, side_length, tt)])])
        moi_xy = sum(
            [sum(a) for a in zip([a * b * c * d / 12 for a, b, c, d in zip(xd, yd, side_length, tt)],
                                 [(a - xc) * (b - yc) * c * d for a, b, c, d in
                                  zip(xm, ym, side_length, tt)])])

        if abs(moi_xy / area ** 2) < 1e-12:
            moi_xy = 0

        # Calculate angle of principal axes
        if moi_xx == moi_yy:
            theta_principal = np.pi / 2
        else:
            theta_principal = np.arctan(
                (-2 * moi_xy) / (moi_xx - moi_yy)) / 2

        # Change to centroid principal coordinates
        # coord12 = [[a - xc for a in cs_sketch.nodes[0]],
        #            [a - yc for a in cs_sketch.nodes[1]]]
        coord12 = np.array([[np.cos(theta_principal), np.sin(theta_principal)],
                            [-np.sin(theta_principal), np.cos(theta_principal)]]).dot(cs_sketch.nodes)

        # re-calculate cross sectional properties for the centroid
        for i in range(nele):
            sn = cs_sketch.elem[0][i]
            fn = cs_sketch.elem[1][i]
            # calculate the coordinate of the mid point of the element
            xm = xm + [mean_list([coord12[0][sn], coord12[0][fn]])]
            ym = ym + [mean_list([coord12[1][sn], coord12[1][fn]])]
            # calculate the dimension of the element
            xd = xd + [(coord12[0][fn] - coord12[0][sn])]
            yd = yd + [(coord12[1][fn] - coord12[1][sn])]

        # calculate the principal moment of inertia
        moi_1 = sum([sum(a) for a in zip([a ** 2 * b * c / 12 for a, b, c in zip(yd, side_length, tt)],
                                         [(a - yc) ** 2 * b * c for a, b, c in
                                          zip(ym, side_length, tt)])])
        moi_2 = sum([sum(a) for a in zip([a ** 2 * b * c / 12 for a, b, c in zip(xd, side_length, tt)],
                                         [(a - xc) ** 2 * b * c for a, b, c in
                                          zip(xm, side_length, tt)])])

        return cls(
            area=area,
            xc=xc,
            yc=yc,
            moi_xx=moi_xx,
            moi_yy=moi_yy,
            moi_xy=moi_xy,
            theta_principal=theta_principal,
            moi_1=moi_1,
            moi_2=moi_2
        )


# TODO: check the plastic table values. expand the library
class Material:
    """
    Material properties.

    Parameters
    ----------
    e_modulus : float
        Modulus of elasticity.
    poisson : float
        Poisson's ratio.
    f_yield : float
        Yield stress
    plasticity : tuple
        Plasticity table (tuple of stress-plastic strain pairs).
        By default, no plasticity is considered.

    """

    def __init__(self, e_modulus, poisson, f_yield, plasticity=None):
        self.e_modulus = e_modulus
        self.poisson = poisson
        self.f_yield = f_yield
        self.plasticity = plasticity
        self.epsilon = np.sqrt(235. / f_yield)

    @staticmethod
    def plastic_table(nominal=None):
        """
        Plasticity tables.

        Returns a tuple with plastic stress-strain curve values for different steels
        given a steel name, e.g 'S355'

        Parameters
        ----------
        nominal : string [optional]
            Steel name. Default value, 'S355'

        Attributes
        ----------

        Notes
        -----

        References
        ----------

        """
        if nominal == None:
            nominal = 'S235'

        if nominal == 'S355':
            table = (
                (355.0, 0.0),
                (360.0, 0.015),
                (390.0, 0.0228),
                (420.0, 0.0315),
                (440.0, 0.0393),
                (480.0, 0.0614),
                (520.0, 0.0926),
                (550.0, 0.1328),
                (570.0, 0.1746),
                (585.0, 0.2216),
                (586.0, 1.)
            )

        if nominal == 'S381':
            table = (
                (381.1, 0.0),
                (391.2, 0.0053),
                (404.8, 0.0197),
                (418.0, 0.0228),
                (444.2, 0.0310),
                (499.8, 0.0503),
                (539.1, 0.0764),
                (562.1, 0.1009),
                (584.6, 0.1221),
                (594.4, 0.1394),
                (596, 1.)
            )

        if nominal == 'S650':
            table = (
                (760., 0.0),
                (770., 0.022),
                (850., 0.075),
                (900., 0.1),
                (901., 1.)
            )

        if nominal == 'S700':
            table = (
                (300., 0.00000),
                (400., 4.53e-5),
                (450., 8.24e-5),
                (500., 1.41e-4),
                (550., 2.47e-4),
                (600., 4.45e-4),
                (630., 6.52e-4),
                (660., 1.00e-3),
                (700., 2.07e-3),
                (720., 3.37e-3),
                (750., 9.18e-3),
                (770., 1.41e-2),
                (790., 2.01e-2),
                (800., 2.38e-2),
                (820., 3.26e-2),
                (840., 4.56e-2),
                (850., 5.52e-2),
                (860., 7.11e-2),
                (865., 9.34e-2),
                (870., 1.00)
            )

        return table

    @classmethod
    def from_nominal(cls, nominal_strength=None):

        """
        Alternative constructor creating a steel material from a given nominal strength.

        Parameters
        ----------

        nominal_strength : str
            Steel quality, given in the form of e.g. "S355"

        """
        if nominal_strength is None:
            f_yield = 235.
        else:
            f_yield = float(nominal_strength.replace('S', ''))

        plasticity = cls.plastic_table(nominal=nominal_strength)
        return cls(210000., 0.3, f_yield, plasticity=plasticity)


class BCs:
    def __init__(self, bcs):
        self.bcs = bcs

    @classmethod
    def from_hinged(cls):
        return cls([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]])


class StructProps:
    """
    Structural properties of a member.

    Parameters
    ----------
    t_classification : float, optional
        Classification of a tube, d/(t^2*e)
    p_classification : float, optional
        Classification of a plate, c/(t*e)
    lmbda_y : float, optional
        Flexural slenderness on the strong axis.
    lmbda_z : float, optional
        Flexural slenderness on the weak axis.
    n_pl_rd : float, optional
        Plastic axial compression resistance.
    n_b_rd_shell : float, optional
        Shell buckling resistance
    """

    def __init__(self,
                 t_classification=None,
                 p_classification=None,
                 lmbda_y=None,
                 lmbda_z=None,
                 n_cr_plate=None,
                 sigma_cr_plate=None,
                 n_pl_rd=None,
                 n_b_rd_plate=None,
                 sigma_b_rd_plate=None,
                 sigma_cr_shell=None,
                 sigma_cr_shell_new=None,
                 lenca=None,
                 lenca_new=None,
                 n_cr_shell=None,
                 n_cr_shell_new=None,
                 sigma_b_rd_shell=None,
                 sigma_b_rd_shell_new=None,
                 n_b_rd_shell=None,
                 n_b_rd_shell_new=None
                 ):
        self.t_classification = t_classification
        self.p_classification = p_classification
        self.lmbda_y = lmbda_y
        self.lmbda_z = lmbda_z
        self.n_cr_plate = n_cr_plate
        self.sigma_cr_plate = sigma_cr_plate
        self.n_pl_rd = n_pl_rd
        self.n_b_rd_plate = n_b_rd_plate
        self.sigma_b_rd_plate = sigma_b_rd_plate
        self.sigma_cr_shell = sigma_cr_shell
        self.sigma_cr_shell_new = sigma_cr_shell_new
        self.lenca = lenca
        self.lenca_new = lenca_new
        self.n_cr_shell = n_cr_shell
        self.n_cr_shell_new = n_cr_shell_new
        self.sigma_b_rd_shell = sigma_b_rd_shell
        self.sigma_b_rd_shell_new = sigma_b_rd_shell_new
        self.n_b_rd_shell = n_b_rd_shell
        self.n_b_rd_shell_new = n_b_rd_shell_new


class Part:
    """
    Structural part.

    Class describing a structural part, including geometry, boundary conditions loads and resistance.

    Parameters
    ----------
    geometry : Geometry object, optional
    cs_props : CsProps object, optional
    material : Material object, optional
    struct_props : StructProps object, optional
    bc_loads: BCs object, optional

    """

    def __init__(self,
                 geometry=None,
                 cs_props=None,
                 material=None,
                 struct_props=None,
                 bc_loads=None
                 ):
        self.geometry = geometry
        self.cs_props = cs_props
        self.material = material
        self.bc_loads = bc_loads
        self.struct_props = struct_props


class SimplySupportedPlate:
    """

    """
    def __init__(self, width, thickness, length, f_y, psi=None):
        if psi is None:
            psi = 1

        self.width = width
        self.thickness = thickness
        self.length = length
        self.f_y = f_y
        self.psi = psi
        self.area = width * thickness

        self.plate_class = plate_class(thickness, width, f_y)
        self.sigma_cr = sigma_cr_plate(thickness, width, psi=psi)
        self.a_eff = calc_a_eff(thickness, width, f_y, psi)


# SIMPLY SUPPORTED PLATE

#TODO: Implement EN50341. Currently the resistance is calculated only for pure compression elements. Add interaction.
def plate_class(
        thickness,
        width,
        f_yield
):
    # Docstring
    """
    Plate classification.

    Returns the class for a given plate, according to EN1993-1-1.
    Currently works for simply supported plates under pure compression.

    Parameters
    ----------
    thickness : float
        [mm] Plate thickness
    width : float
        [mm] Plate width
    f_yield : float
        [MPa] Yield stress

    Returns
    -------
    int
        [_] Class number

    Notes
    -----
    To be extended to include the rest of the cases of Table 5.3 [1].
    Members under combined axial and bending and outstand members.

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings. Brussels: CEN, 2005

    """

    # Convert inputs to floats
    width, thickness, f_yield = float(width), float(thickness), float(f_yield)

    # Calculate classification
    classification = width / (thickness * np.sqrt(235. / f_yield))
    if classification <= 33.01:
        p_class = 1
    elif classification <= 38.01:
        p_class = 2
    elif classification <= 42.01:
        p_class = 3
    else:
        p_class = 4

    # Return value
    return p_class


def plate_class_new(
        thickness,
        width,
        f_yield
):
    # Docstring
    """
    Plate classification acc.to final draft of EC3-1-1.

    Returns the class for a given plate, according to EN1993-1-1.
    Currently works for simply supported plates under pure compression.

    Parameters
    ----------
    thickness : float
        [mm] Plate thickness
    width : float
        [mm] Plate width
    f_yield : float
        [MPa] Yield stress

    Returns
    -------
    int
        [_] Class number

    Notes
    -----
    To be extended to include the rest of the cases of Table 5.3 [1].
    Members under combined axial and bending and outstand members.

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings. Brussels: CEN, 2005

    """

    # Convert inputs to floats
    width, thickness, f_yield = float(width), float(thickness), float(f_yield)

    # Calculate classification
    classification = width / (thickness * np.sqrt(235. / f_yield))
    if classification <= 28.01:
        p_class = 1
    elif classification <= 34.01:
        p_class = 2
    elif classification <= 38.01:
        p_class = 3
    else:
        p_class = 4

    # Return value
    return p_class


def sigma_cr_plate(
        thickness,
        width,
        psi=None
):
    # Docstring
    """
    Critical stress of a plate.

    Calculates the critical stress for a simply supported plate.

    Parameters
    ----------
    thickness : float
        [mm] Plate thickness
    width : float
        [mm] Plate width
    psi : float, optional
        [_] Ratio of the min over max stress for a linear distribution,
        (sigma_min / sigma_max)
        Default = 1, which implies a uniform distribution

    Returns
    -------
    float
        [MPa] Plate critical stress

    Notes
    -----
    To be extended to include cantilever plate (outstand members)

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-5: Plated structural elements. Brussels: CEN, 2005.

    """
    # Convert inputs to floats
    thickness, width = float(thickness), float(width)

    # Default value for psi
    if psi is None:
        psi = 1.
    else:
        psi = float(psi)

    # Calculate kapa_sigma
    k_sigma = 8.2 / (1.05 + psi)

    # Elastic critical stress acc. to EN3-1-5 Annex A
    sigma_e = 190000 * (thickness / width) ** 2
    sigma_cr = sigma_e * k_sigma

    # Return value
    return sigma_cr


def calc_a_eff(
        thickness,
        width,
        f_yield,
        psi=None):
    # Docstring
    """
    Plastic design resistance of a plate.

    Calculates the resistance of a plate according to EN1993-1-1 and
    EN1993-1-5. The plate is assumed simply supported.

    Parameters
    ----------
    thickness : float
        [mm] Plate thickness
    width : float
        [mm] Plate width
    f_yield : float
        [MPa] Yield stress
    psi : float, optional
        [_] Ratio of the min over max stress for a linear distribution,
        (sigma_min / sigma_max)
        Default = 1, which implies a uniform distribution

    Returns
    -------
    float
        [N] Plastic design resistance

    Notes
    -----
    To be extended to include cantilever plate (outstand members)

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings.
        Brussels: CEN, 2005.
    .. [2] Eurocode 3: Design of steel structures - Part 1-5: Plated structural elements. Brussels: CEN, 2005.

    """

    # Convert inputs to floats
    thickness, width, f_yield = float(thickness), float(width), float(f_yield)

    # Default value for psi
    if psi is None:
        psi = 1.
    else:
        psi = float(psi)

    # Calculate kapa_sigma
    k_sigma = 8.2 / (1.05 + psi)

    # Aeff calculation.
    # Reduction factor for the effective area of the profile acc. to EC3-1-5
    classification = width / (thickness * np.sqrt(235. / f_yield))
    lambda_p = classification / (28.4 * np.sqrt(k_sigma))
    if lambda_p > 0.673 and plate_class(thickness, width, f_yield) == 4:
        rho = (lambda_p - 0.055 * (3 + psi)) / lambda_p ** 2
    else:
        rho = 1.

    # Effective area
    a_eff = rho * thickness * width

    return(a_eff)


def calc_a_eff_new(
        thickness,
        width,
        f_yield,
        psi=None):
    # Docstring
    """
    Plastic design resistance of a plate.

    Calculates the resistance of a plate according to EN1993-1-1 and
    EN1993-1-5. The plate is assumed simply supported.

    Parameters
    ----------
    thickness : float
        [mm] Plate thickness
    width : float
        [mm] Plate width
    f_yield : float
        [MPa] Yield stress
    psi : float, optional
        [_] Ratio of the min over max stress for a linear distribution,
        (sigma_min / sigma_max)
        Default = 1, which implies a uniform distribution

    Returns
    -------
    float
        [N] Plastic design resistance

    Notes
    -----
    To be extended to include cantilever plate (outstand members)

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings.
        Brussels: CEN, 2005.
    .. [2] Eurocode 3: Design of steel structures - Part 1-5: Plated structural elements. Brussels: CEN, 2005.

    """

    # Convert inputs to floats
    thickness, width, f_yield = float(thickness), float(width), float(f_yield)

    # Default value for psi
    if psi is None:
        psi = 1.
    else:
        psi = float(psi)

    # Calculate kapa_sigma
    k_sigma = 8.2 / (1.05 + psi)

    # Aeff calculation.
    # Reduction factor for the effective area of the profile acc. to EC3-1-5
    classification = width / (thickness * np.sqrt(235. / f_yield))
    lambda_p = classification / (28.4 * np.sqrt(k_sigma))
    if lambda_p > 0.673 and plate_class_new(thickness, width, f_yield) == 4:
        rho = (lambda_p - 0.055 * (3 + psi)) / lambda_p ** 2
    else:
        rho = 1.

    # Effective area
    a_eff = rho * thickness * width

    return(a_eff)


# CYLINDRICAL SHELLS

def tube_class(
        thickness,
        radius,
        f_yield
):
    """
    CHS classification.

    Returns the class for a given plate, according to EN1993-1-1.
    Currently works for simply supported plates under pure compression.

    Parameters
    ----------
    thickness : float
        [mm] Plate thickness
    radius : float
        [mm] Tube radius
    f_yield : float
        [MPa] Yield stress

    Returns
    -------
    int
        [_] Class number

    Notes
    -----
    To be extended to include the rest of the cases of Table 5.3 [1].
    Members under combined axial and bending and outstand members.

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings. Brussels: CEN, 2005

    """

    # Convert inputs to floats
    radius, thickness, f_yield = float(radius), float(thickness), float(f_yield)

    # Calculate classification
    classification = 2 * radius / (thickness * np.sqrt(235. / f_yield)**2)
    if classification <= 50.01:
        t_class = 1
    elif classification <= 70.01:
        t_class = 2
    elif classification <= 90.01:
        t_class = 3
    else:
        t_class = 4

    # Return value
    return t_class


def shell_length_category(
        radius,
        thickness,
        length
):
    """Return the length gategory of a cylinder acc. to EC3-1-6 D.1.2.1"""

    omega = length / np.sqrt(radius * thickness)
    if 1.7 <= omega <= 0.5 * (radius / thickness):
        length_category = 1
    elif omega < 1.7:
        length_category = 0
    else:
        length_category = 2

    return length_category


def shell_length_category_new(
        radius,
        thickness,
        length
):
    """Return the length gategory of a cylinder acc. to EC3-1-6 D.1.2.1"""

    omega = length / np.sqrt(radius * thickness)
    if 1.7 <= omega <= 2.86 * (radius / thickness):
        length_category = 1
    elif omega < 1.7:
        length_category = 0
    else:
        length_category = 2

    return length_category


def sigma_x_rcr(
        thickness,
        radius,
        length,
        kapa_bc=None,
        e_modulus=None
):
    """
    Critical meridional stress for cylindrical shell.

    Calculates the critical load for a cylindrical shell under pure
    compression and assumes uniform stress distribution. Calculation
    according to EN1993-1-6 [1], Annex D.

    Parameters
    ----------
    thickness : float
        [mm] Shell thickness
    radius : float
        [mm] Cylinder radius
    length : float
        [mm] Cylnder length

    Returns
    -------
    list
        List of 2 elements:
        a) float, Critical load [N]
        b) string, length category

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-6: Strength and stability of shell structures.
        Brussels: CEN, 2006.

    """

    if kapa_bc is None:
        kapa_bc = 1

    if e_modulus is None:
        e_modulus = 210000.

    # Convert inputs to floats
    thickness, radius, length = float(thickness), float(radius), float(length)

    # Length category
    lenca = shell_length_category(radius, thickness, length)

    # Elastic critical load acc. to EN3-1-6 Annex D
    omega = length / np.sqrt(radius * thickness)
    if lenca == 1:
        c_x = 1.
    elif lenca == 0:
        c_x = 1.36 - (1.83 / omega) + (2.07 / omega ** 2)
    elif lenca == 2:
        # c_x_b is read on table D.1 of EN3-1-5 Annex D acc. to BCs
        # BC1 - BC1 is used on the Abaqus models (both ends clamped, see EN3-1-5 table 5.1)
        c_x_b = 6.
        c_x_n = max((1 + 0.2 * (1 - 2 * omega * thickness / radius) / c_x_b), 0.6)
        c_x = c_x_n

    # Calculate critical stress, eq. D.2 on EN3-1-5 D.1.2.1-5
    sigma_cr = 0.605 * 210000 * c_x * thickness / radius

    return sigma_cr


def sigma_x_rcr_new(
        thickness,
        radius,
        length,
        kapa_bc=None,
        e_modulus=None
):
    """
    Critical meridional stress for cylindrical shell.

    Calculates the critical load for a cylindrical shell under pure
    compression and assumes uniform stress distribution. Calculation
    according to EN1993-1-6 [1], Annex D.

    Parameters
    ----------
    thickness : float
        [mm] Shell thickness
    radius : float
        [mm] Cylinder radius
    length : float
        [mm] Cylnder length

    Returns
    -------
    list
        List of 2 elements:
        a) float, Critical load [N]
        b) string, length category

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-6: Strength and stability of shell structures.
        Brussels: CEN, 2006.

    """

    if kapa_bc is None:
        kapa_bc = 1

    if e_modulus is None:
        e_modulus = 210000.

    # Convert inputs to floats
    thickness, radius, length = float(thickness), float(radius), float(length)

    lenca = shell_length_category_new(radius, thickness, length)
    omega = length / np.sqrt(radius * thickness)

    if lenca == 0:
        c_x = 1.36 - (1.83 / omega) + (2.07 / omega ** 2)
        # Calculate critical stress, eq. D.2 on EN3-1-5 D.1.2.1-5
        sigma_cr = 0.605 * 210000 * c_x * thickness / radius
    elif lenca == 1:
        c_x = 1.
        # Calculate critical stress, eq. D.2 on EN3-1-5 D.1.2.1-5
        sigma_cr = 0.605 * 210000 * c_x * thickness / radius
    elif lenca == 2:
        # flex buckling
        r_o = radius + thickness / 2.
        r_i = radius - thickness / 2.
        moi = np.pi * (r_o ** 4 - r_i ** 4) / 4.
        area = 2 * np.pi * radius * thickness
        sigma_cr = n_cr_flex(length, moi, kapa_bc=kapa_bc, e_modulus=e_modulus) / area
    else:
        print("Wrong length category.")

    # Return value
    return sigma_cr


def n_cr_shell(
        thickness,
        radius,
        length
):
    """
    Critical compressive load for cylindrical shell.

    Calculates the critical load for a cylindrical shell under pure
    compression and assumes uniform stress distribution. Calculation
    according to EN1993-1-6 [1], Annex D.

    Parameters
    ----------
    thickness : float
        [mm] Shell thickness
    radius : float
        [mm] Cylinder radius
    length : float
        [mm] Cylnder length

    Returns
    -------
    float
        [N] Critical load

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-6: Strength and stability of shell structures.
        Brussels: CEN, 2006.

    """

    # Convert inputs to floats
    thickness, radius, length = float(thickness), float(radius), float(length)

    # Elastic critical load acc to EN3-1-6 Annex D
    nn_cr_shell = 2 * np.pi * radius * thickness * sigma_x_rcr(thickness, radius, length)

    # Return value
    return nn_cr_shell


def n_cr_shell_new(
        thickness,
        radius,
        length
):
    """
    Critical compressive load for cylindrical shell.

    Calculates the critical load for a cylindrical shell under pure
    compression and assumes uniform stress distribution. Calculation
    according to EN1993-1-6 [1], Annex D.

    Parameters
    ----------
    thickness : float
        [mm] Shell thickness
    radius : float
        [mm] Cylinder radius
    length : float
        [mm] Cylnder length

    Returns
    -------
    float
        [N] Critical load

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-6: Strength and stability of shell structures.
        Brussels: CEN, 2006.

    """

    # Convert inputs to floats
    thickness, radius, length = float(thickness), float(radius), float(length)

    # Elastic critical load acc to EN3-1-6 Annex D
    nn_cr_shell = 2 * np.pi * radius * thickness * sigma_x_rcr_new(thickness, radius, length)

    # Return value
    return nn_cr_shell


def sigma_x_rk(
        thickness,
        radius,
        length,
        f_y_k,
        fab_quality=None,
        flex_kapa=None
):
    """
    Meridional design buckling stress.

    Calculates the meridional buckling stress for a cylindrical shell
    according to EN1993-1-6 [1].

    Parameters
    ----------
    thickness : float
        [mm] Shell thickness
    radius : float
        [mm] Cylinder radius
    length : float
        [mm] Cylnder length
    f_y_k : float
        [MPa] Characteristic yield strength
    fab_quality : str, optional
        [_] Fabrication quality class. Accepts: 'fcA', 'fcB', 'fcC'
        The three classes correspond to .006, .010 and .016 times the
        width of a dimple on the shell.
        Default = 'fcA', which implies excelent fabrication
    gamma_m1 : int, optional
        [_] Partial safety factor
        Default = 1.1

    Returns
    -------
    float
        [MPa] Meridional buckling stress

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-6: Strength and stability of shell structures.
        Brussels: CEN, 2006._

    """

    # Default values
    if fab_quality is None:
        fab_quality = 'fcA'

    if flex_kapa is None:
        flex_kapa = 1.
        
    # Check class acc. to EC3-1
    t_class = tube_class(thickness, radius, f_y_k)
    if t_class == 4:
        # Fabrication quality class acc. to table D2
        if fab_quality is 'fcA':
            q_factor = 40.
        elif fab_quality is 'fcB':
            q_factor = 25.
        elif fab_quality is 'fcC':
            q_factor = 16.
        else:
            print('Invalid fabrication class input. Choose between \'fcA\', \'fcB\' and \'fcC\' ')
            return
    
        # Critical meridional stress, calculated on separate function
        category = shell_length_category(radius, thickness, length)
        sigma_cr = sigma_x_rcr(thickness, radius, length)
    
        # Shell slenderness
        lmda = np.sqrt(f_y_k / sigma_cr)
        delta_w_k = (1. / q_factor) * np.sqrt(radius / thickness) * thickness
        alpha = 0.62 / (1 + 1.91 * (delta_w_k / thickness) ** 1.44)
        beta = 0.6
        eta = 1.
        if category == 2:
            # For long cylinders, a formula is suggested for lambda, EC3-1-6 D1.2.2(4)
            # Currently, the general form is used. to be fixed.
            lmda_0 = 0.2
            # lmda_0 = 0.2 + 0.1 * (sigma_e_M / sigma_e)
        else:
            lmda_0 = 0.2
    
        lmda_p = np.sqrt(alpha / (1. - beta))
    
        # Buckling reduction factor, chi
        if lmda <= lmda_0:
            chi_shell = 1.
        elif lmda < lmda_p:
            chi_shell = 1. - beta * ((lmda - lmda_0) / (lmda_p - lmda_0)) ** eta
        else:
            chi_shell = alpha / (lmda ** 2)
    
        # Buckling stress
        sigma_rk = chi_shell * f_y_k
        sigma_rd_shell = sigma_rk
    else:
        sigma_rd_shell = f_y_k
    
    # flex buckling
    r_o = radius + thickness / 2.
    r_i = radius - thickness / 2.
    moi = np.pi * (r_o ** 4 - r_i ** 4) / 4.
    area = 2 * np.pi * radius * thickness
    chi = chi_flex(length, area, moi, f_y_k, b_curve="c", kapa_bc=flex_kapa)
    sigma_rd = sigma_rd_shell * chi

    # Return value
    return sigma_rd


def sigma_x_rk_new(
        thickness,
        radius,
        length,
        f_y_k,
        fab_quality=None,
        flex_kapa=None
):
    """
    Meridional characteristic buckling stress.

    Calculates the characteristic meridional buckling stress for a cylindrical shell according to EN1993-1-6 [1].

    Parameters
    ----------
    thickness : float
        [mm] Shell thickness
    radius : float
        [mm] Cylinder radius
    length : float
        [mm] Cylnder length
    f_y_k : float
        [MPa] Characteristic yield strength
    fab_quality : str, optional
        [_] Fabrication quality class. Accepts: 'fcA', 'fcB', 'fcC'
        The three classes correspond to .006, .010 and .016 times the
        width of a dimple on the shell.
        Default = 'fcA', which implies excelent fabrication
    gamma_m1 : int, optional
        [_] Partial safety factor
        Default = 1.2 (new suggestion from prEN draft)

    Returns
    -------
    float
        [MPa] Meridional buckling stress

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-6: Strength and stability of shell structures.
        Brussels: CEN, 2006._

    """

    # Default values
    if fab_quality is None:
        fab_quality = 'fcA'

    if flex_kapa is None:
        flex_kapa = 1.

    t_class = tube_class(thickness, radius, f_y_k)
    
    if t_class == 4:
        # Fabrication quality class acc. to table D2
        if fab_quality is 'fcA':
            q_factor = 40.
        elif fab_quality is 'fcB':
            q_factor = 25.
        elif fab_quality is 'fcC':
            q_factor = 16.
        else:
            print('Invalid fabrication class input. Choose between \'fcA\', \'fcB\' and \'fcC\' ')
            return
    
        # Critical meridinal stress, calculated on separate function
        category = shell_length_category_new(radius, thickness, length)
        sigma_cr = sigma_x_rcr_new(thickness, radius, length)
    
        # Shell slenderness
        lmda = np.sqrt(f_y_k / sigma_cr)
        delta_w_k = (1. / q_factor) * np.sqrt(radius / thickness) * thickness
        alpha_xG = 0.83
        alpha_I = 0.06 + 0.93 / (1 + 2.7 * (delta_w_k / thickness)**0.85)
        alpha = alpha_xG * alpha_I
        
        #alpha = 0.62 / (1 + 1.91 * (delta_w_k / thickness) ** 1.44)
        beta = 1 - 0.95 / (1 + 1.12 * (delta_w_k / thickness))
        eta = 5.8 / (1 + 4.6 * (delta_w_k / thickness)**0.87)
        chi_xh = 1.15
        lambda_p = np.sqrt(alpha / (1 - beta))
        
        if category == 2:
            # For long cylinders, a formula is suggested for lambda under compression/bending , EC3-1-6 D1.2.2(4)
            # Pure compression is assumed.
            lmda_0 = 0.1 + (0 / 1.)
            # lmda_0 = 0.2 + 0.1 * (sigma_e_M / sigma_e)
        else:
            lmda_0 = 0.2
    
        lmda_p = np.sqrt(alpha / (1. - beta))
    
        # Buckling reduction factor, chi
        if lmda <= lmda_0:
            chi_shell = chi_xh - (lmda / lmda_0)*(chi_xh - 1)
        elif lmda < lmda_p:
            chi_shell = 1. - beta * ((lmda - lmda_0) / (lmda_p - lmda_0)) ** eta
        else:
            chi_shell = alpha / (lmda ** 2)
    
        # Buckling stress
        sigma_rk = chi_shell * f_y_k
        sigma_rd_shell = sigma_rk / gamma_m1
    else:
        sigma_rd_shell = f_y_k

    # flex buckling
    r_o = radius + thickness / 2.
    r_i = radius - thickness / 2.
    moi = np.pi * (r_o ** 4 - r_i ** 4) / 4.
    area = 2 * np.pi * radius * thickness
    chi = chi_flex(length, area, moi, f_y_k, b_curve="c", kapa_bc=flex_kapa)
    sigma_rd = sigma_rd_shell * chi

    # Return value
    return sigma_rd


def sigma_x_rd(
        thickness,
        radius,
        length,
        f_y_k,
        fab_quality=None,
        gamma_m1=None,
        flex_kapa=None
):
    """
    Meridional design buckling stress.

    Calculates the meridional buckling stress for a cylindrical shell
    according to EN 1993-1-6 [1]. Flexural buckling is also checked acc to EN 1993-1-1 [2]

    Parameters
    ----------
    thickness : float
        [mm] Shell thickness
    radius : float
        [mm] Cylinder radius
    length : float
        [mm] Cylnder length
    f_y_k : float
        [MPa] Characteristic yield strength
    fab_quality : str, optional
        [_] Fabrication quality class. Accepts: 'fcA', 'fcB', 'fcC'
        The three classes correspond to .006, .010 and .016 times the
        width of a dimple on the shell.
        Default = 'fcA', which implies excelent fabrication
    gamma_m1 : int, optional
        [_] Partial safety factor
        Default = 1.1

    Returns
    -------
    float
        [MPa] Meridional buckling stress

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-6: Strength and stability of shell structures.
        Brussels: CEN, 2006._
    .. [2] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings.
        Brussels: CEN, 2005._

    """

    # Default values
    if fab_quality is None:
        fab_quality = 'fcA'

    if gamma_m1 is None:
        # The default value 1.2 is as the lowest recommended on EC3-1-6 8.5.2(10) NOTE.
        gamma_m1 = 1.1
    else:
        gamma_m1 = float(gamma_m1)

    if flex_kapa is None:
        flex_kapa = 1.

    sigma_xx_rd = sigma_x_rk(
        thickness,
        radius,
        length,
        f_y_k,
        fab_quality=fab_quality,
        flex_kapa=flex_kapa
    ) / gamma_m1

    return sigma_xx_rd


def sigma_x_rd_new(
        thickness,
        radius,
        length,
        f_y_k,
        fab_quality=None,
        gamma_m1=None,
        flex_kapa=None
):
    """
    Meridional design buckling stress.

    Calculates the meridional buckling stress for a cylindrical shell
    according to the new draft of EN1993-1-6 [1].

    Parameters
    ----------
    thickness : float
        [mm] Shell thickness
    radius : float
        [mm] Cylinder radius
    length : float
        [mm] Cylnder length
    f_y_k : float
        [MPa] Characteristic yield strength
    fab_quality : str, optional
        [_] Fabrication quality class. Accepts: 'fcA', 'fcB', 'fcC'
        The three classes correspond to .006, .010 and .016 times the
        width of a dimple on the shell.
        Default = 'fcA', which implies excelent fabrication
    gamma_m1 : int, optional
        [_] Partial safety factor
        Default = 1.1

    Returns
    -------
    float
        [MPa] Meridional buckling stress

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-6 draft: Strength and stability of shell structures.
        Brussels: CEN, 2006._

    """

    # Default values
    if fab_quality is None:
        fab_quality = 'fcA'

    if gamma_m1 is None:
        # The default value 1.2 is as the lowest recommended on EC3-1-6 8.5.2(10) NOTE.
        gamma_m1 = 1.2
    else:
        gamma_m1 = float(gamma_m1)

    if flex_kapa is None:
        flex_kapa = 1.

    sigma_xx_rd = sigma_x_rk(
        thickness,
        radius,
        length,
        f_y_k,
        fab_quality=fab_quality,
        flex_kapa=flex_kapa
    ) / gamma_m1

    return sigma_xx_rd


# OVERALL BUCKLING
def n_cr_flex(
        length,
        moi_y,
        kapa_bc=None,
        e_modulus=None
):
    # Docstring
    """
    Euler's critical load.

    Calculates the critical load for flexural buckling of a given column.
    A single direction is considered. If more directions are required
    (e.g the two principal axes), the function has to be called multiple
    times. For torsional mode critical load use n_cr_tor(), and for
    flexural-torsional critical load use n_cr_flex_tor()

    Parameters
    ----------
    length : float
        [mm] Column length.
    moi_y : float
        [mm^4] Moment of inertia.
    kapa_bc : float, optional
        [_] length correction for the effect of the boundary conditions.
        Default = 1, which implies simply supported column.
    e_modulus : float, optional
        [MPa] Modulus of elasticity.
        Default = 210000., typical value for steel.

    Returns
    -------
    float
        [N] Critical load.

    """
    # default values
    if kapa_bc is None:
        kapa_bc = 1.
    else:
        kapa_bc = float(kapa_bc)

    if e_modulus is None:
        e_modulus = 210000.
    else:
        e_modulus = float(e_modulus)

    # Euler's critical load
    nn_cr_flex = (np.pi ** 2) * e_modulus * moi_y / (kapa_bc * length) ** 2

    # Return the result
    return nn_cr_flex


def n_cr_tor(
        length,
        area,
        moi_y0,
        moi_z0,
        moi_torsion,
        moi_warp,
        y_0=None,
        z_0=None,
        e_modulus=None,
        poisson=None,
):
    # Docstring
    """
    Torsional elastic critical load

    Calculates the torsional elastic critical load for a hinged column.
    The input values are refering to the principal axes. For flexural
    buckling (Euler cases) use n_cr_flex. For the combined
    flexural-torsional modes use n_cr_flex_tor.

    Parameters
    ----------
    length : float
        [mm] Column length.
    area : float
        [mm^2] Cross-sectional area.
    moi_y0 : float
        [mm^4] Moment of inertia around `y`-axis.
        `y`-axis on the centre of gravity but not necessarily principal.
    moi_z0 : float
        [mm^4] Moment of inertia around `z`-axis.
        `z`-axis on the centre of gravity but not necessarily principal.
    moi_torsion : float
        [mm^4] Saint Venant constant.
    moi_warp : float
        [mm^6] Torsion constant.
    y_0 : float, optional
        [mm] Distance on `y`-axis of the shear center to the origin.
        Default = 0, which implies symmetric profile
    z_0 : float, optional
        [mm] Distance on `z`-axis of the shear center to the origin.
        Default = 0, which implies symmetric profile
    e_modulus : float, optional
        [MPa] Modulus of elasticity.
        Default = 210000., general steel.
    poisson : float, optional
        [_] Young's modulus of elasticity.
        Default = 0.3, general steel.

    Returns
    -------
    float
        [N] Flexural-torsional critical load.

    Notes
    -----
    The torsional critical load is calculated as:

    .. math:: N_{cr, tor} = {GJ + {\pi^2EI_w\over{L^2}}\over{r^2}}

    Where:
        :math:`E`    : Elasticity modulus

        :math:`G`    : Shear modulus

        :math:`J`    : Torsional constant (Saint Venant)

        :math:`I_w`  : Warping constant

        :math:`r^2=(moi_y + moi_z)/A + x_0^2 + y_0^2`

        :math:`x_0, y_0`  : Shear centre coordinates on the principal coordinate system


    References
    ----------
    ..[1]N. S. Trahair, Flexural-torsional buckling of structures, vol. 6. CRC Press, 1993.
    ..[2]NS. Trahair, MA. Bradford, DA. Nethercot, and L. Gardner, The behaviour and design of steel structures to EC3, 4th edition. London; New York: Taylor & Francis, 2008.

    """
    # default values
    if y_0 is None:
        y_0 = 0
    else:
        y_0 = float(y_0)

    if z_0 is None:
        z_0 = 0
    else:
        z_0 = float(z_0)

    if e_modulus is None:
        e_modulus = 210000.
    else:
        e_modulus = float(e_modulus)

    if poisson is None:
        poisson = 0.3
    else:
        poisson = float(poisson)

    # Shear modulus
    g_modulus = e_modulus / (2 * (1 + poisson))

    # Polar radius of gyration.
    i_pol = np.sqrt((moi_y0 + moi_z0) / area)
    moi_zero = np.sqrt(i_pol ** 2 + y_0 ** 2 + z_0 ** 2)

    # Calculation of critical torsional load.
    nn_cr_tor = (1 / moi_zero ** 2) * (g_modulus * moi_torsion + (np.pi ** 2 * e_modulus * moi_warp / length ** 2))

    # Return the result
    return nn_cr_tor


def n_cr_flex_tor(
        length,
        area,
        moi_y,
        moi_z,
        moi_yz,
        moi_torsion,
        moi_warp,
        y_sc=None,
        z_sc=None,
        e_modulus=None,
        poisson=None,
):
    # Docstring
    """
    Flexural-Torsional elastic critical load

    Calculates the critical load for flexural-torsional buckling of a
    column with hinged ends. The returned value is the minimum of the
    the three flexural-torsional and the indepedent torsional mode, as
    dictated in EN1993-1-1 6.3.1.4 [1]. (for further details, see Notes).

    Parameters
    ----------
    length : float
        [mm] Column length.
    area : float
        [mm^2] Cross-sectional area.
    moi_y : float
        [mm^4] Moment of inertia around `y`-axis.
        `y`-axis on the centre of gravity but not necessarily principal.
    moi_z : float
        [mm^4] Moment of inertia around `z`-axis.
        `z`-axis on the centre of gravity but not necessarily principal.
    moi_yz : float
        [mm^4] Product of inertia.
    moi_torsion : float
        [mm^4] Saint Venant constant.
    moi_warp : float
        [mm^6] Torsion constant.
    y_sc : float, optional
        [mm] Distance on `y`-axis of the shear center to the origin.
        Default = 0, which implies symmetric profile
    z_sc : float, optional
        [mm] Distance on `z`-axis of the shear center to the origin.
        Default = 0, which implies symmetric profile
    e_modulus : float, optional
        [MPa] Modulus of elasticity.
        Default = 210000., general steel.
    poisson : float, optional
        [_] Young's modulus of elasticity.
        Default = 0.3, general steel.

    Returns
    -------
    float
        [N] Flexural-torsional critical load.

    Notes
    -----
    The flexural-torsional critical loads are calculated as a combination
    of the three independent overall buckling modes:
    i)   flexural around the major axis,
    ii)  flexural around the minor axis,
    iii) Torsional buckling (around x-axis).

    First, the cs-properties are described on the principal axes. Then
    the three independent  modes are calculated. The combined
    flexural-torsional modes are calculated as the roots of a 3rd order
    equation, as given in [1], [2]. The minimum of the torsional and the
    three combined modes is returned (the two independent flexural modes
    are not considered; for critical load of pure flexural mode use
    'n_cr_flex').

    References
    ----------
    ..[1]N. S. Trahair, Flexural-torsional buckling of structures, vol. 6. CRC Press, 1993.
    ..[2]NS. Trahair, MA. Bradford, DA. Nethercot, and L. Gardner, The behaviour and design of steel structures to EC3, 4th edition. London; New York: Taylor & Francis, 2008.

    """
    # default values
    if y_sc is None:
        y_sc = 0
    else:
        y_sc = float(y_sc)

    if z_sc is None:
        z_sc = 0
    else:
        z_sc = float(z_sc)

    if e_modulus is None:
        e_modulus = 210000.
    else:
        e_modulus = float(e_modulus)

    if poisson is None:
        poisson = 0.3
    else:
        poisson = float(poisson)

    # Angle of principal axes
    if abs(moi_y - moi_z) < 1e-20:
        theta = np.pi / 4
    else:
        theta = -np.arctan((2 * moi_yz) / (moi_y - moi_z)) / 2

    # Distance of the rotation centre to the gravity centre on the
    # principal axes coordinate system
    y_0 = y_sc * np.cos(-theta) - z_sc * np.sin(-theta)
    z_0 = z_sc * np.cos(-theta) + y_sc * np.sin(-theta)

    # Moment of inertia around principal axes.
    moi_y0 = (moi_y + moi_z) / 2 + np.sqrt(((moi_y - moi_z) / 2) ** 2 + moi_yz ** 2)
    moi_z0 = (moi_y + moi_z) / 2 - np.sqrt(((moi_y - moi_z) / 2) ** 2 + moi_yz ** 2)

    # Polar radius of gyration.
    i_pol = np.sqrt((moi_y0 + moi_z0) / area)
    moi_zero = np.sqrt(i_pol ** 2 + y_0 ** 2 + z_0 ** 2)

    # Independent critical loads for flexural and torsional modes.
    n_cr_max = (np.pi ** 2 * e_modulus * moi_y0) / (length ** 2)
    n_cr_min = (np.pi ** 2 * e_modulus * moi_z0) / (length ** 2)
    n_tor = n_cr_tor(
        length,
        area,
        moi_y0,
        moi_z0,
        moi_torsion,
        moi_warp=moi_warp,
        y_0=y_0,
        z_0=z_0,
        e_modulus=e_modulus,
        poisson=poisson
    )

    # Coefficients of the 3rd order equation for the critical loads
    # The equation is in the form aaaa * N ^ 3 - bbbb * N ^ 2 + cccc * N - dddd
    aaaa = moi_zero ** 2 - y_0 ** 2 - z_0 ** 2
    bbbb = ((n_cr_max + n_cr_min + n_tor) * moi_zero ** 2) - (n_cr_min * y_0 ** 2) - (n_cr_max * z_0 ** 2)
    cccc = moi_zero ** 2 * (n_cr_min * n_cr_max) + (n_cr_min * n_tor) + (n_tor * n_cr_max)
    dddd = moi_zero ** 2 * n_cr_min * n_cr_max * n_tor

    det_3 = (
        4 * (-bbbb ** 2 + 3 * aaaa * cccc) ** 3 + (2 * bbbb ** 3 - 9 * aaaa * bbbb * cccc + 27 * aaaa ** 2 * dddd) ** 2
    )

    if det_3 < 0:
        det_3 = -1. * det_3
        cf = 1j
    else:
        cf = 1

    # Critical load
    # The following n_cr formulas are the roots of the 3rd order equation of the global critical load

    n_cr_1 = bbbb / (3. * aaaa) - (2 ** (1. / 3) * (-bbbb ** 2 + 3 * aaaa * cccc)) / \
                                  (3. * aaaa * (2 * bbbb ** 3 - 9 * aaaa * bbbb * cccc + 27 * aaaa ** 2 * dddd + \
                                                (cf * np.sqrt(det_3))) ** (1. / 3)) + (
                                                                                   2 * bbbb ** 3 - 9 * aaaa * bbbb * cccc + 27 * aaaa ** 2 * dddd + \
                                                                                   (cf * np.sqrt(det_3))) ** (1. / 3) / (
                                                                                   3. * 2 ** (1. / 3) * aaaa)

    n_cr_2 = bbbb / (3. * aaaa) + ((1 + (0 + 1j) * np.sqrt(3)) * (-bbbb ** 2 + 3 * aaaa * cccc)) / \
                                  (3. * 2 ** (2. / 3) * aaaa * (
                                  2 * bbbb ** 3 - 9 * aaaa * bbbb * cccc + 27 * aaaa ** 2 * dddd + \
                                  (cf * np.sqrt(det_3))) ** (1. / 3)) - ((1 - (0 + 1j) * np.sqrt(3)) * \
                                                                      (
                                                                      2 * bbbb ** 3 - 9 * aaaa * bbbb * cccc + 27 * aaaa ** 2 * dddd + \
                                                                      (cf * np.sqrt(det_3))) ** (1. / 3)) / (
                                                                     6. * 2 ** (1. / 3) * aaaa)

    n_cr_3 = bbbb / (3. * aaaa) + ((1 - (0 + 1j) * np.sqrt(3)) * (-bbbb ** 2 + 3 * aaaa * cccc)) / \
                                  (3. * 2 ** (2. / 3) * aaaa * (
                                  2 * bbbb ** 3 - 9 * aaaa * bbbb * cccc + 27 * aaaa ** 2 * dddd + \
                                  (cf * np.sqrt(det_3))) ** (1. / 3)) - ((1 + (0 + 1j) * np.sqrt(3)) * \
                                                                      (
                                                                      2 * bbbb ** 3 - 9 * aaaa * bbbb * cccc + 27 * aaaa ** 2 * dddd + \
                                                                      (cf * np.sqrt(det_3))) ** (1. / 3)) / (
                                                                     6. * 2 ** (1. / 3) * aaaa)

    # Lowest root is the critical load
    nn_cr_flex_tor = min(abs(n_cr_1), abs(n_cr_2), abs(n_cr_3), n_tor)

    # Return the critical load
    return nn_cr_flex_tor


def lmbda_flex(
        length,
        area,
        moi_y,
        f_yield,
        kapa_bc=None,
        e_modulus=None,
):
    # Docstring
    """
    Flexural slenderness.

    Calculates the slenderness of a columne under pure compression.
    Euler's critical load is used.

    Parameters
    ----------
    length : float
        [mm] Column length
    area : float
        [mm^2] Cross section area
    moi_y : float
        [mm^4] Moment of inertia
    kapa_bc : float, optional
        [_] length correction for the effect of the boundary conditions.
        Default = 1, which implies simply supported column
    e_modulus : float, optional
        [MPa] Modulus of elasticity
        Default = 210000., typical value for steel
    f_yield : float, optional
        [MPa] yield stress.
        Default = 380., brcause this value was used extencively while the
        function was being written. To be changed to 235.

    Returns
    -------
    float
        [_] Member slenderness

    """
    # default values
    if kapa_bc is None:
        kapa_bc = 1.
    else:
        kapa_bc = float(kapa_bc)

    if e_modulus is None:
        e_modulus = 210000.
    else:
        e_modulus = float(e_modulus)

    if f_yield is None:
        f_yield = 235.
    else:
        f_yield = float(f_yield)

    # Calculate Euler's critical load
    n_cr = n_cr_flex(
        length,
        moi_y,
        e_modulus=e_modulus,
        kapa_bc=kapa_bc
    )

    # Flexural slenderness EN3-1-1 6.3.1.3 (1)
    lmbda_flexx = np.sqrt(area * f_yield / n_cr)

    # Return the result
    return lmbda_flexx


def imp_factor(b_curve):
    # Docstring
    """
    Imperfection factor.

    Returns the imperfection factor for a given buckling curve.
    The values are taken from Table 6.1 of EN1993-1-1 [1]

    Parameters
    ----------
    b_curve : {'a0', 'a', 'b', 'c', 'd'}
        [_] Name of the buckling curve as obtained from Table 6.2 of [1].

    Returns
    -------
    float
        [_] Imperfection factor.

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings.
        Brussels: CEN, 2005.

    """
    switcher = {
        'a0': 0.13,
        'a': 0.21,
        'b': 0.34,
        'c': 0.49,
        'd': 0.76,
    }
    return switcher.get(b_curve, "nothing")


def chi_flex(
        length,
        area,
        moi_y,
        f_yield,
        b_curve,
        kapa_bc=None
):
    # Docstring
    """
    Flexural buckling reduction factor.

    Claculates the reduction factor, chi, according to EN1993-1-1 6.3.1.2

    Parameters
    ----------
    length : float
        [mm] Column length
    area : float
        [mm^2] Cross section area
    moi_y : float
        [mm^4] Moment of inertia
    f_yield : float
        [MPa] Yield stress.
    b_curve : str
        [_] Name of the buckling curve as obtained from Table 6.2 of [1].
        Valid options are {'a0', 'a', 'b', 'c', 'd'}
    kapa_bc : float, optional
        [_] length correction for the effect of the boundary conditions.
        Default = 1, which implies simply supported column

    Returns
    -------
    float
        [_] Reduction factor.

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings.
        Brussels: CEN, 2005.

    """
    if kapa_bc is None:
        kapa_bc = 1.

    lmda = lmbda_flex(
        length,
        area,
        moi_y,
        f_yield,
        kapa_bc=kapa_bc,
        e_modulus=None,
    )
    if lmda < 0.2:
        chi = 1.
    else:
        alpha = imp_factor(b_curve)

        phi = (1 + alpha * (lmda - 0.2) + lmda ** 2) / 2.

        chi = 1 / (phi + np.sqrt(phi ** 2 - lmda ** 2))

        if chi > 1.:
            chi = 1.

    return chi


def n_b_rd(
        length,
        area,
        moi_y,
        f_yield,
        b_curve,
        kapa_bc=None,
        gamma_m1=None
):
    # Docstring
    """
    Flexural buckling resistance.

    Verifies the resistance of a column against flexural buckling
    according to EN1993-1-1 6.3.1.1.

    Parameters
    ----------
    length : float
        [mm] Column length
    area : float
        [mm^2] Cross section area
    moi_y : float
        [mm^4] Moment of inertia
    f_yield : float
        [MPa] Yield stress.
    b_curve : str
        [_] Name of the buckling curve as obtained from Table 6.2 of [1].
        Valid options are: {'a0', 'a', 'b', 'c', 'd'}
    kapa_bc : float, optional
        [_] Length correction for the effect of the boundary conditions.
        Default = 1, which implies simply supported column
    gamma_m1 : float, optional
        [_] Partial safety factor.
        Default = 1.

    Returns
    -------
    float
        [N] Buckling resistance.

    References
    ----------
    .. [1] Eurocode 3: Design of steel structures - Part 1-1: General rules and rules for buildings.
        Brussels: CEN, 2005.

    """
    if kapa_bc is None:
        kapa_bc = 1.

    if gamma_m1 is None:
        gamma_m1 = 1.

    chi = chi_flex(length,
                   area,
                   moi_y,
                   f_yield,
                   b_curve,
                   kapa_bc=kapa_bc)

    nn_b_rd = area * f_yield * chi / gamma_m1

    return nn_b_rd


# CONNECTIONS
def bolt_grade2stress(bolt_grade):
    # Docstring
    """
    Convert bolt grade to yield and ultimate stress.

    Standard designation for bolt grade as a decimal is converted to yield and ultimate stress values in MPa. In the
    standard bolt grade designation, the integer part of the number represents the ultimate stress in MPa/100 and the
    decimal part is the yield stress as a percentage of the ultimate (e.g 4.6 is f_u = 400, f_y = 400 * 0.6 = 240).

    Parameters
    ----------
    bolt_grade : float

    Returns
    -------
    tuple : (f_ultimate, f_yield)
    """
    # Calculation using divmod
    f_ultimate = 100 * divmod(bolt_grade, 1)[0]
    f_yield = round(f_ultimate * divmod(bolt_grade, 1)[1])

    # Return values
    return f_ultimate, f_yield


def shear_area(bolt_size, shear_threaded=None):
    # Docstring
    """
    Shear area of a bolt.

    Returns the srea to be used for the calculation of shear resistance of a bolt, either the gross cross-section of the
    bolt (circle area) or the reduced area of the threaded part of the bolt.

    Parameters
    ----------
    bolt_size : float
        Bolt's diameter.
    shear_threaded : bool, optional
        Designates if the shear plane is on the threaded portion or not.
        Default in False, which implies shearing of the non-threaded portion

    Returns
    -------
    float

    Notes
    -----
    Currently, the threaded area is based on an average reduction of the shank area. To be changed to analytic formula.
    """
    # Default
    if shear_threaded is None:
        shear_threaded = False

    # Calculate area
    if shear_threaded:
        a_shear = 0.784 * (np.pi * bolt_size ** 2 / 4)
    else:
        a_shear = np.pi * bolt_size ** 2 / 4

    # Return
    return a_shear


def f_v_rd(
        bolt_size,
        bolt_grade,
        shear_threaded=None,
        gamma_m2=None
):
    # Docstring
    """
    Bolt's shear resistance.

    Calculates the shear resistance of single bolt for one shear plane as given in table 3.4 of EC3-1-8.

    Parameters
    ----------
    bolt_size : float
        Diameter of the non-threaded part (nominal bolt size e.g. M16 = 16)
    bolt_grade : float
        Bolt grade in standard designation format (see documentation of bolt_grade2stress())
    shear_threaded : bool, optional
        Designates if the shear plane is on the threaded portion or not.
        Default in False, which implies shearing of the non-threaded portion
    gamma_m2 : float, optional
        Safety factor.
        Default value is 1.25

    Returns
    -------
    float

    """

    # Defaults
    bolt_size = float(bolt_size)
    if shear_threaded is None:
        shear_threaded = False

    if gamma_m2 is None:
        gamma_m2 = 1.25
    else:
        gamma_m2 = float(gamma_m2)

    # av coefficient
    if shear_threaded and bolt_grade == (4.6 or 8.6):
        a_v = 0.5
    else:
        a_v = 0.6

    # Get ultimate stress for bolt
    f_ub = bolt_grade2stress(bolt_grade)[0]

    # Shear area
    a_shear = shear_area(bolt_size, shear_threaded)

    # Shear resistance
    ff_v_rd = a_v * f_ub * a_shear / gamma_m2

    # Return value
    return ff_v_rd


def bolt_min_dist(d_0):
    """
    Minimum bolt spacing.

    :param d_0:
    :return:
    """
    e_1 = 1.2 * d_0
    e_2 = 1.2 * d_0
    e_3 = 1.5 * d_0
    p_1 = 2.2 * d_0
    p_2 = 2.4 * d_0

    return e_1, e_2, e_3, p_1, p_2


def f_b_rd(bolt_size, bolt_grade, thickness, steel_grade, f_yield, distances, d_0):
    """
    Connection bearing capacity.

    Calculates the bearing capacity of a single bolt on a plate. The distances to the plate edges/other bolts are
    described
    :param bolt_size:
    :param bolt_grade:
    :param thickness:
    :param steel_grade:
    :param f_yield:
    :param distances:
    :param d_0:
    :return:
    """
    pass


def f_weld_perp():
    # f_w_1 = (sqrt(2) / 2) * a_weld * l_weld * f_ult / (b_w * gamma_m2)
    # f_w_2 = 0.9 * f_ult * a_weld * l_weld * sqrt(2) / gamma_m2
    pass


def f_weld_paral():
    pass


def bolt2washer(m_bolt):
    """
    Washer diameter.

    Return the diameter of the washer for a given bolt diameter.
    The calculation is based on a function derived from linear regression
    on ENXXXXXXX[REF].

    Parameters
    ----------
    m_bolt : float
        Bolt diameter

    """

    d_washer = np.ceil(1.5893 * m_bolt + 5.1071)
    return d_washer


def fabclass_2_umax(fab_class):
    """
    Maximum displacement for a given fabrication class acc. to EC3-1-6.

    Parameters
    ----------
    fab_class : {"fcA", "fcB", "fcC"}

    """
    # Assign imperfection amplitude, u_max acc. to the fabrication class
    if fab_class is 'fcA':
        u_max = 0.006
    elif fab_class is 'fcB':
        u_max = 0.010
    else:
        u_max = 0.016

    # Return values
    return u_max


def mean_list(numbers):
    """
    Mean value.

    Calculate the average for a list of numbers.

    Parameters
    ----------
    numbers : list

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """

    return float(sum(numbers)) / max(len(numbers), 1)
