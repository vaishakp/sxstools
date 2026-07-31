import numpy as np
from waveformtools.waveformtools import message, unsort
from waveformtools.grids import GLGrid
from waveformtools.diagnostics import MethodInfo as method_info
from spectools.spherical.swsh import Yslm_vec, Yslm_prec
from waveformtools.single_mode import SingleMode
from spectools.chebyshev.chebyshev import ChebyshevSpectral
import sxstools.rotations as rotations

import re


def ToSphericalPolar(coords, centres=[0, 0, 0]):
    """Transform the given cartesian coordinate points to
    spherical polar form.

    Parameters
    ----------
    coords : 3darray
             An array of time series of the
             3D Cartesian coords. The ordering
             of the axis is (x, y, z)
    centres : 2darray
              An array containing the timeseries
              of the coordinate center of the AH.
              The axis order is (time, [xf, yf, zf])

    Returns
    -------
    sphp_coords : 4darray

    """
    # print("Corrected transform")

    x, y, z = coords

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    x_com, y_com, z_com = centres

    x -= x_com
    y -= y_com
    z -= z_com

    # x = (x.T - x_com).T
    # y = (y.T - y_com).T
    # z = (z.T - z_com).T
    # r_eq = np.sqrt(x**2 + y**2)

    radius = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arccos(z / radius)

    phi = np.arctan2(y, x) % (2 * np.pi)

    return [radius, theta, phi]


def ToCartesian(coords):
    """Transform the given cartesian coordinate points to
    spherical polar form"""

    r, theta, phi = coords

    r_eq = r * np.sin(theta)

    x = r_eq * np.cos(phi)
    y = r_eq * np.sin(phi)
    z = r * np.cos(theta)

    return [x, y, z]


def GetDomainRadii(file_name):
    """Get the radii of the domains given
    the `GrDomain.input` file

    Parameters
    ----------
    file_name : str
                The file name

    Returns
    -------
    radii : dict
            A dictionary containing the
            radial extents of the domain.
    """

    line_no = 0

    radii = {}

    with open(file_name, "r") as fd:

        while True:

            line_no += 1
            line = fd.readline()

            if not line:
                message("Reached EOF", message_verbosity=3)
                break

            elif "Sphere" in line:
                # print('Sphere found in line', line)
                # elif 1:
                # print(line)

                # print('Printing regex')
                # res = re.match('Sphere\w\d\s*=\(\s*Radii\s*=', line)
                res = re.search("Sphere\w\s*=\(Radii\s*=\s*", line)

                if res is not None:
                    # if 1:

                    message(f"Line found at {line_no}", message_verbosity=3)
                    message(line, message_verbosity=3)
                    message("Regex res", res, message_verbosity=3)

                    s1, s2 = res.span()

                    sphere = line[s1 : s1 + 7]

                    res_ref = re.split(",", line[s2:])

                    # res_ref = [float(item.removesuffix(';\n')) for item in res_ref]

                    res_ref[-1] = res_ref[-1][:-2]

                    res_ref = [float(item) for item in res_ref]

                    # print(res_ref[-1])
                    radii.update({sphere: res_ref})

                    message("Split res", message_verbosity=3)
                    message(res_ref, message_verbosity=3)

            else:
                pass

    return radii


def GetSphereRadialExtents(all_radii, sub_domain="SphereA0"):
    """Get the radial extents of a subdomain

    Parameters
    ----------
    all_radii : dict of strings
                The dictionary containing all the
                radii of the spheres.
    sub_domain : str
                 A string indicating the subdomain
                 of interest.
    """

    subdomain_class = sub_domain[:-1]
    sphere_num = int(sub_domain[-1])

    subd_radii = all_radii[subdomain_class]

    r1 = subd_radii[sphere_num]
    r2 = subd_radii[sphere_num + 1]

    return r1, r2


def VolumeDataCartesianToSpectralParallel(cartesian_vol_data, r1=-1, r2=1):
    """Transform the given volume data in cartesian coordinates to
    spectral SH x CS space via decomposition, in parallel.

    Parameters
    ----------
    cartesian_vol_data : 3darray of floats
                         The volume data on cartesian grid
    ell_max : int
              The max SH basis function order to use
              for angular decompositions.
    n_max : int
            The max number of Chebyshev polynomials
            to use for radial decompositions.

    Returns
    -------
    spectral_vol_data : 3darray of floats
                        The volume data on spectral space.

    """

    n_r, n_theta, n_phi = cartesian_vol_data.shape

    message("Num of radial points", n_r, message_verbosity=2)

    grid_info = GLGrid(L=n_theta - 1)

    # message(f"grid info ell max {grid_info.ell_max}")
    message("L grid", grid_info.L, message_verbosity=2)

    minfo = method_info(ell_max=grid_info.L, int_method="GL")
    message(f"method info ell max {minfo.ell_max}", message_verbosity=2)

    # Radial set of Clm modes.
    # Each element is a singlemode obj
    modes_r_set = []
    # Construct Clm(r).
    # SHExpand at every radial shell
    diag = True

    extract = True
    from qlmtools.transforms import SHExpand, SHContract

    r_indices = np.arange(n_r)

    for r_index in range(n_r):

        ang_data = cartesian_vol_data[r_index, :, :]

        one_set_modes = SHExpand(func=ang_data, method_info=minfo, info=grid_info)

        if extract:
            ext_modes = one_set_modes
            extract = False

        message(f"one set modes ell max {one_set_modes.ell_max}", message_verbosity=4)
        message("Modes dict", one_set_modes.get_modes_dict(), message_verbosity=4)

        modes_r_set.append(one_set_modes)

        if diag:
            ang_data_recon = SHContract(
                modes=one_set_modes.get_modes_dict(),
                info=grid_info,
                ell_max=grid_info.L,
            )

            delta = np.sqrt(np.mean((ang_data - ang_data_recon) ** 2))
            message("RMS deviation", delta, message_verbosity=3)

    message(
        "Modes before r decomp 0 l2 m0", modes_r_set[0].mode(2, 0), message_verbosity=4
    )

    # Construct PClm
    from chebyshev import ChebyshevSpectral

    cs = ChebyshevSpectral(a=r1, b=r2, Nfuncs=n_r)

    message(
        f"Created Chebyshev radial grid with Nfuncs {cs.Nfuncs}\n",
        f"Shape of collocation points \
            {cs.collocation_points_logical.shape}",
        message_verbosity=3,
    )

    modes_Clmq = SingleMode(ell_max=minfo.ell_max, modes_dim=n_r)

    # modes_Clmq = {}

    for ell in range(minfo.ell_max + 1):

        this_ell_modes = {}

        for emm in range(-ell, ell + 1):

            this_r_modes = get_radial_clms(modes_r_set, ell, emm)
            this_Clmq = cs.MatrixPhysToSpec @ np.array(this_r_modes)
            message(f" This Clmq l{ell}, m{emm}", this_Clmq, message_verbosity=4)

            if ell == 2 and emm == 0:
                this_r_modes_recon = cs.MatrixSpecToPhys @ this_Clmq

                delta = np.mean((this_r_modes_recon - this_r_modes) ** 2)
                message(
                    f"Spectral diagnostics l{ell} m{emm} Delta",
                    delta,
                    message_verbosity=4,
                )

            # this_ell_modes.update({f'm{emm}' : this_Clmq})
            modes_Clmq.set_mode_data(ell, emm, this_Clmq)
        # modes_Clmq.update({f'l{ell}' : this_ell_modes})

    return modes_Clmq, ext_modes


def VolumeDataCartesianToSpectral(cartesian_vol_data, r1=-1, r2=1):
    """Transform the given volume data in cartesian coordinates to
    spectral SH x CS space via decomposition.

    Parameters
    ----------
    cartesian_vol_data : 3darray of floats
                         The volume data on cartesian grid
    ell_max : int
              The max SH basis function order to use
              for angular decompositions.
    n_max : int
            The max number of Chebyshev polynomials
            to use for radial decompositions.

    Returns
    -------
    spectral_vol_data : 3darray of floats
                        The volume data on spectral space.

    """

    n_r, n_theta, n_phi = cartesian_vol_data.shape

    message("Num of radial points", n_r, message_verbosity=3)

    grid_info = GLGrid(L=n_theta - 1)

    # message(f"grid info ell max {grid_info.ell_max}")
    message("L grid", grid_info.L, message_verbosity=3)

    minfo = method_info(ell_max=grid_info.L, int_method="GL")
    message(f"method info ell max {minfo.ell_max}", message_verbosity=3)

    # Radial set of Clm modes.
    # Each element is a singlemode obj
    modes_r_set = []
    # Construct Clm(r).
    # SHExpand at every radial shell
    diag = True

    extract = True
    from qlmtools.transforms import SHExpand, SHContract

    for r_index in range(n_r):

        ang_data = cartesian_vol_data[r_index, :, :]
        one_set_modes = SHExpand(func=ang_data, method_info=minfo, info=grid_info)

        if extract:
            ext_modes = one_set_modes
            extract = False

        message(f"one set modes ell max {one_set_modes.ell_max}", message_verbosity=4)
        message("Modes dict", one_set_modes.get_modes_dict(), message_verbosity=4)
        modes_r_set.append(one_set_modes)

        if diag:
            ang_data_recon = SHContract(
                modes=one_set_modes.get_modes_dict(),
                info=grid_info,
                ell_max=grid_info.L,
            )

            delta = np.sqrt(np.mean((ang_data - ang_data_recon) ** 2))
            message("RMS deviation", delta, message_verbosity=3)

    message(
        "Modes before r decomp 0 l2 m0", modes_r_set[0].mode(2, 0), message_verbosity=4
    )

    # Construct PClm

    cs = ChebyshevSpectral(a=r1, b=r2, Nfuncs=n_r)

    message(
        f"Created Chebyshev radial grid with Nfuncs {cs.Nfuncs}\n",
        f"Shape of collocation points \
            {cs.collocation_points_logical.shape}",
        message_verbosity=3,
    )

    modes_Clmq = SingleMode(ell_max=minfo.ell_max, modes_dim=n_r)

    # modes_Clmq = {}

    for ell in range(minfo.ell_max + 1):

        this_ell_modes = {}

        for emm in range(-ell, ell + 1):

            this_r_modes = get_radial_clms(modes_r_set, ell, emm)

            this_Clmq = cs.MatrixPhysToSpec @ np.array(this_r_modes)

            message(f" This Clmq l{ell}, m{emm}", this_Clmq, message_verbosity=4)

            if ell == 2 and emm == 0:
                this_r_modes_recon = cs.MatrixSpecToPhys @ this_Clmq

                delta = np.mean((this_r_modes_recon - this_r_modes) ** 2)
                message(
                    f"Spectral diagnostics l{ell} m{emm} Delta",
                    delta,
                    message_verbosity=4,
                )

            # this_ell_modes.update({f'm{emm}' : this_Clmq})
            modes_Clmq.set_mode_data(ell, emm, this_Clmq)

        # modes_Clmq.update({f'l{ell}' : this_ell_modes})

    return modes_Clmq, ext_modes


def get_radial_clms(modes_set, ell, emm):
    """Get the radial set of modes Clms corresponding to
    one lm from a radial set of all Clms"""

    n_r = len(modes_set)

    clm_r_set = []

    for r_index in range(n_r):

        clm_r_set.append(modes_set[r_index].mode(ell, emm))

    return np.array(clm_r_set)


def evaluate_Clmq_on_grid_backup(
    modes_Clmq, cart_coords=None, sphp_coords=None, centers=None, Nfuncs=8
):
    """Find the values of the function in spectral space represented
    by its modes `modes_Clmq` on a requested set of coordinates .

    Parameters
    ----------
    modes_Clmq : dict
                 The modes dictionary with keys in the
                 format `lxny` and values being the radial
                 spectral Cehnyshev coefficients for that
                 mode.
    cart_coords : list of three 3darrays, optional
                  The meshgrid `X,Y,Z` of cartesian coordinates onto
                  which to evaluate the function.

    centers : list
              The coordinate center of the cartesian
              coodinate arrays.

    sphp_coords : list of three 3darrays, optional
                  The meshgrid `r, theta, phi` of
                  spherical polar coordinates onto
                  which to evaluate the function.
    Returns
    -------
    func_vals : 2darray
                The value of the function on the requested
                cartesian gridpoints.
    """

    if np.array(sphp_coords).all() == np.array(None):

        X, Y, Z = cart_coords

        if np.array(centers).all() == np.array(None):
            centers = [0, 0, 0]

        xcom, ycom, zcom = centers
        message("Transforming to spherical polar coordinates", message_verbosity=3)

        sphp_coords = ToSphericalPolar(cart_coords, centers)

    R, Th, Ph = sphp_coords

    Rf = R.flatten()
    Th = Th.flatten()
    Ph = Ph.flatten()

    arg_order = np.argsort(Rf)

    Rf = Rf[arg_order]
    Th = Th[arg_order]
    Ph = Ph[arg_order]

    set_of_coords = ReorganizeCoords([Rf, Th, Ph])

    message("Set of coords", message_verbosity=4)

    for item in set_of_coords:
        message(item[0].shape, message_verbosity=4)

    # from itertools import zip_longest

    index = 0

    for item in set_of_coords:

        Rf, Th, Ph = item

        # grid_prod = zip_longest(Rf, Th, Ph)

        all_ells = list(modes_Clmq.keys())
        all_ells = [int(item[1:]) for item in all_ells]

        # For every radial grid point, evaluate the angular part

        func_vals = np.zeros(sphp_coords[0].shape, dtype=complex).flatten()

        prev_ri = Rf[0]

        prev_Clm_interp = RContract(modes_Clmq, prev_ri, r1, r2, Nfuncs)

        for ri, thi, phi in grid_prod:
            index += 1
            if ri != prev_ri:

                message("Re interpolating", message_verbosity=3)
                prev_ri = ri

                prev_Clm_interp = RContract(modes_Clmq, prev_ri, r1, r2, Nfuncs)

                message("Reint done", message_verbosity=2)

            func_vals[index] = AngContract(prev_Clm_interp, thi, phi)

            message(index / len(Rf), message_verbosity=4)

    return func_vals.reshape(sphp_coords[0].shape)


def evaluate_Clmq_on_grid(
    modes_Clmq, r1, r2, cart_coords=None, sphp_coords=None, centers=None, Nfuncs=8
):
    """Find the values of the function in spectral space represented
    by its modes `modes_Clmq` on a requested set of coordinates .

    Parameters
    ----------
    modes_Clmq : dict
                 The modes dictionary with keys in the
                 format `lxny` and values being the radial
                 spectral Cehnyshev coefficients for that
                 mode.
    cart_coords : list of three 3darrays, optional
                  The meshgrid `X,Y,Z` of cartesian coordinates onto
                  which to evaluate the function.

    centers : list
              The coordinate center of the cartesian
              coodinate arrays.

    sphp_coords : list of three 3darrays, optional
                  The meshgrid `r, theta, phi` of
                  spherical polar coordinates onto
                  which to evaluate the function.
    Returns
    -------
    func_vals : 2darray
                The value of the function on the requested
                cartesian gridpoints.
    """

    if np.array(sphp_coords).all() == np.array(None):

        X, Y, Z = cart_coords

        if np.array(centers).all() == np.array(None):
            centers = [0, 0, 0]

        xcom, ycom, zcom = centers
        message("Transforming to spherical polar coordinates", message_verbosity=3)

        sphp_coords = ToSphericalPolar(cart_coords, centers)

    R, Th, Ph = sphp_coords

    set_of_coords = ReorganizeCoords(sphp_coords)

    message("Set of coords", message_verbosity=4)

    for item in set_of_coords:
        message(item[0].shape, message_verbosity=4)

    # Rf = R.flatten()
    # Th = Th.flatten()
    # Ph = Ph.flatten()

    # arg_order = np.argsort(Rf)

    # Rf = Rf[arg_order]
    # Th = Th[arg_order]
    # Ph = Ph[arg_order]

    # from itertools import zip_longest

    func_vals_list = []
    index = 0

    # prev_ri, _, _ = set_of_coords[0]
    message(R.shape, message_verbosity=4)
    prev_ri = np.amin(R)

    prev_Clm_interp = RContract(modes_Clmq, prev_ri, r1, r2, Nfuncs)

    message("prev Clm", prev_Clm_interp.mode(2, 0), message_verbosity=4)

    ext_mode = prev_Clm_interp

    for item in set_of_coords:

        Rf, Th, Ph = item
        ri = Rf[0]

        message("At radius", ri, message_verbosity=4)

        index += 1

        if ri != prev_ri:

            message("Re r interpolating", message_verbosity=3)
            prev_ri = ri

            prev_Clm_interp = RContract(modes_Clmq, prev_ri, r1, r2, Nfuncs)
        message("Reint r done", message_verbosity=2)

        func_vals_list += list(AngContract(prev_Clm_interp, Th, Ph))

        message(index / len(set_of_coords), message_verbosity=2)

        # grid_prod = zip_longest(Rf, Th, Ph)

        # all_ells = list(modes_Clmq.keys())
        # all_ells = [int(item[1:]) for item in all_ells]

        # For every radial grid point, evaluate the angular part

        # func_vals = np.zeros(sphp_coords[0].shape, dtype=complex).flatten()

        # prev_ri = Rf[0]

        # prev_Clm_interp = RContract(modes_Clmq, prev_ri, r1, r2, Nfuncs)

        # index = 0

        # for ri, thi, phi in grid_prod:

        func_vals = np.array(func_vals_list)

    return func_vals.reshape(sphp_coords[0].shape), ext_mode


def ReorganizeCoords(sphp_coords):
    """Reorganize the spherical polar coords
    into a list of vector arrays"""

    # Original any 2d mesh
    R, Th, Ph = sphp_coords

    # Flatten
    Rf = np.array(R).flatten()
    Th = np.array(Th).flatten()
    Ph = np.array(Ph).flatten()

    # Sort as per increasing radius
    arg_order = np.argsort(Rf)

    Rf = Rf[arg_order]
    Th = Th[arg_order]
    Ph = Ph[arg_order]

    # Compute change in R
    rdiff = np.diff(Rf)

    message("Rdiff", rdiff, message_verbosity=4)

    # Compute the change locations of R shells
    change_locs = np.where(rdiff > 0)[0]

    if len(change_locs) > 0:
        if change_locs[0] == 0:
            message(
                "Change in the very first entry." "Ignoring empty entry...",
                message_verbosity=4,
            )

            change_locs = change_locs[1:]
    # Add it it the last element. This is to ensure
    # that the last element is included when slicing
    # below

    change_locs = np.array(list(change_locs) + [len(Rf)])

    message("Change locs", change_locs, message_verbosity=4)

    # List og coord groups
    list_of_coord_groups = []
    prev_index = 0

    count = 0
    for index in change_locs:

        thisRf = Rf[prev_index:index]
        thisTh = Th[prev_index:index]
        thisPh = Ph[prev_index:index]

        # message(f"Group {count}", message_verbosity=4)
        count += 1

        # message(f" R {thisRf}\n Th {thisTh} \n Ph {thisPh}", message_verbosity=4)
        # message(" \n ------------------------- \n",
        # message_verbosity=4)

        list_of_coord_groups.append([thisRf, thisTh, thisPh])

        prev_index = index

    return list_of_coord_groups, arg_order


def GetEllMaxFromModesDict(modes_Clmq):
    """Get the `ell_max` from modes dict."""

    all_ells = list(modes_Clmq.keys())
    all_ells = [int(item[1:]) for item in all_ells]

    ell_max = max(all_ells)

    return ell_max, all_ells


def RContractSerial(modes_Clmq, r_value, r1=None, r2=None, Nfuncs=None, cs=None):
    """Return the Clm modes at the requested radius

    Parameters
    ----------
    modes_Clmq : SingleMode obj
                 The 3D spectral mode coefficients
    r_value : float
              The radius of the sphere onto which
              the Clm coefficients need to be computed
    r1, r2 : float
             The radii of the ends of the subdomain
             i.e. of the physical radial collocation
             points
    Nfuncs : int
             The number of basis functions in the
             chebyshev expansion

    Returns
    -------
    Clm_at_r_values : SingleMode obj
                      The modes at the requested
                      radius.
    """

    if cs is None:
        if (r1 is None) or (r2 is None) or (Nfuncs is None):
            raise KeyError(
                "Please provide the Chebyshev basis or"
                "the values of r_min, r_max and Nfuncs"
            )

        else:
            from chebyshev import ChebyshevSpectral

            # from chebyshev_basis import ChebyshevBasis

            cs = ChebyshevSpectral(Nfuncs=Nfuncs, a=r1, b=r2)
            # cb = ChebyshevBasis(Nfuncs=Nfuncs)

    ell_max = modes_Clmq.ell_max

    Clm_at_r_values = SingleMode(ell_max=ell_max)

    message(
        "Created R contract modes data shape",
        Clm_at_r_values._modes_data.shape,
        message_verbosity=4,
    )
    message(
        "Input Clmq modes data shape", modes_Clmq._modes_data.shape, message_verbosity=4
    )

    # print(f"Intercept r value {r_value}")

    for ell in range(ell_max + 1):
        for emm in range(-ell, ell + 1):

            cheb_coeffs = modes_Clmq.mode(ell, emm)

            # this_Clm = np.sum([cheb_coeffs[order_index] * cb.ChebBasisDirect(r_value_coll, order_index)/cbar[order_index] for order_index in range(Nfuncs)])
            # this_Clm = np.sum([cheb_coeffs[order_index] * cb.ChebBasisDirect(r_value_coll, order_index) for order_index in range(cs.Nfuncs)])
            this_Clm = np.sum(
                [
                    cheb_coeffs[order_index] * cs.EvaluateBasis(r_value, order_index)
                    for order_index in range(cs.Nfuncs)
                ]
            )

            this_Clm0 = cs.MatrixSpecToPhys @ cheb_coeffs

            # if ell==2 and emm==0:
            #    message('Mode Value 2 0 direct', this_Clm, message_verbosity=3)
            #    message('Mode Value 2 0 matrix', this_Clm0[0], message_verbosity=3)
            #    message('Diff', this_Clm-this_Clm0[0], message_verbosity=3)
            Clm_at_r_values.set_mode_data(ell, emm, this_Clm)

    return Clm_at_r_values


def RContract(modes_Clmq, r_value, r1=None, r2=None, Nfuncs=None, cs=None, t_step=None):
    """Return the Clm modes at the requested radius

    Parameters
    ----------
    modes_Clmq : SingleMode obj
                 The 3D spectral mode coefficients
    r_value : float
              The radius of the sphere onto which
              the Clm coefficients need to be computed
    r1, r2 : float
             The radii of the ends of the subdomain
             i.e. of the physical radial collocation
             points
    Nfuncs : int
             The number of basis functions in the
             chebyshev expansion

    Returns
    -------
    Clm_at_r_values : SingleMode obj
                      The modes at the requested
                      radius.
    """

    if cs is None:
        if (r1 is None) or (r2 is None) or (Nfuncs is None):
            raise KeyError(
                "Please provide the Chebyshev basis or"
                "the values of r_min, r_max and Nfuncs"
            )

        else:
            from spectral.chebyshev.chebyshev import ChebyshevSpectral

            # from chebyshev_basis import ChebyshevBasis

            cs = ChebyshevSpectral(Nfuncs=Nfuncs, a=r1, b=r2)
            # cb = ChebyshevBasis(Nfuncs=Nfuncs)

    ell_max = modes_Clmq.ell_max

    Clm_at_r_values = SingleMode(ell_max=ell_max)

    message(
        "Created R contract modes data shape",
        Clm_at_r_values._modes_data.shape,
        message_verbosity=3,
    )
    message(
        "Input Clmq modes data shape", modes_Clmq._modes_data.shape, message_verbosity=3
    )

    # print(f"Intercept r value {r_value}")

    # ell+1, 2*ell +1, Nfuncs
    mode_axes_shape = modes_Clmq._modes_data.shape
    n_t_modes = mode_axes_shape[-1]

    if t_step is None:
        message(
            "The interpolator has no time axis."
            " Carrying out single time evaluation.",
            message_verbosity=3,
        )

        modes_data = modes_Clmq._modes_data
    else:

        message(
            "The interpolator is has time axis." " Selecting time step.",
            message_verbosity=3,
        )

        modes_data = modes_Clmq._modes_data[..., t_step]

    message("Mode array shape", modes_Clmq._modes_data.shape, message_verbosity=4)
    message("Thist Modes data shape", modes_data.shape, message_verbosity=4)

    # Nfuncs
    eval_basis_vec = cs.EvaluateBasis(r_value, np.arange(cs.Nfuncs))

    message("eval basis vec shape", eval_basis_vec.shape, message_verbosity=4)

    # Nfuncs
    this_Clm_mat = np.dot(modes_data, eval_basis_vec)

    Clm_at_r_values._modes_data = this_Clm_mat

    message(f"Clm_at_r_values modes shape {this_Clm_mat.shape}", message_verbosity=4)
    if np.isnan(r_value).any():
        raise ValueError("Nan found in RContract r value!!")

    if np.isnan(eval_basis_vec).any():

        message("r_value", r_value)

        message("eval basis vec", eval_basis_vec)

        raise ValueError("Nan found in RContract eval basis vec!!")

    if np.isnan(modes_data).any():
        raise ValueError("Nan found in RContract modes data!!")

    if np.isnan(Clm_at_r_values._modes_data).any():
        raise ValueError("Nan found in RContract!!")

    return Clm_at_r_values


def AngContract_backup(modes_Clm, theta, phi):
    """Return the values of the function at
    the requested angular location `theta, phi`,
    whose spectral angular modes are given.

    Parameters
    ----------
    modes_Clm : modes dict
                A dictionary of modes
    theta, phi : float/2darray
                 The angular coordinates
                 at which to evaluate the
                 modes.

    Returns
    -------
    func_value : float or 2darray
                 The function at the requested
                 angular coordinates.
    """
    message("Ang contracting", message_verbosity=3)

    try:
        sh = theta.shape
        func_value = np.zeros(sh, dtype=np.complex128)
    except Exception as ex:
        message(ex)
        func_value = 0

    # ell_max, all_ells = GetEllMaxFromModesDict(modes_Clm)
    ell_max = modes_Clm.ell_max

    for ell in range(ell_max + 1):
        # for ell_key, ell_dict in modes_Clm.items():
        for emm in range(-ell, ell + 1):
            # ell = int(ell_key[1:])

            # for emm_key, emm_entry in ell_dict.items():
            #    emm = int(emm_key[1:])
            Clm = modes_Clm.mode(ell, emm)
            # func_value += modes_Clm[f'l{ell}'][f'm{emm}']*Yslm_vec(spin_weight=0, ell=ell, emm=emm, theta_grid=theta, phi_grid=phi)
            func_value += Clm * Yslm_vec(
                spin_weight=0, ell=ell, emm=emm, theta_grid=theta, phi_grid=phi
            )

    message("Done AngCont", message_verbosity=3)
    return func_value


def AngContract(modes_Clm, theta, phi):
    """Return the values of the function at
    the requested angular location `theta, phi`,
    whose spectral angular modes are given.

    Parameters
    ----------
    modes_Clm : modes dict
                A dictionary of modes
    theta, phi : float/2darray
                 The angular coordinates
                 at which to evaluate the
                 modes.

    Returns
    -------
    func_value : float or 2darray
                 The function at the requested
                 angular coordinates.
    """
    message("Ang contracting", message_verbosity=3)

    try:
        sh = theta.shape
        message("Creating func value array", message_verbosity=4)
        func_value = np.zeros(sh, dtype=np.complex128)
    except Exception as ex:
        message("Creating float func value variable", message_verbosity=4)
        message(ex)
        func_value = 0

    # ell_max, all_ells = GetEllMaxFromModesDict(modes_Clm)
    ell_max = modes_Clm.ell_max

    for ell in range(ell_max + 1):
        # for ell_key, ell_dict in modes_Clm.items():
        for emm in range(-ell, ell + 1):
            # ell = int(ell_key[1:])

            # for emm_key, emm_entry in ell_dict.items():
            #    emm = int(emm_key[1:])
            Clm = modes_Clm.mode(ell, emm)
            Yval = Yslm_vec(
                spin_weight=0, ell=ell, emm=emm, theta_grid=theta, phi_grid=phi
            )
            # print("Angles ", theta, phi, f"\t ell {ell} emm{emm}", "\tClm", Clm, "\t Yval", Yval)
            # print("Yslm")
            # func_value += modes_Clm[f'l{ell}'][f'm{emm}']*Yslm_vec(spin_weight=0, ell=ell, emm=emm, theta_grid=theta, phi_grid=phi)
            # print("Intercept angles", theta, phi)
            func_value += Clm * Yval
            # func_value += Clm*Yslm_prec(spin_weight=0, ell=ell, emm=emm, theta=theta[0], phi=phi[0])

    message("Done AngCont", message_verbosity=3)
    return func_value


def AngContract2(modes_Clm, theta, phi):
    """Return the values of the function at
    the requested angular location `theta, phi`,
    whose spectral angular modes are given.

    Parameters
    ----------
    modes_Clm : modes dict
                A dictionary of modes
    theta, phi : float/2darray
                 The angular coordinates
                 at which to evaluate the
                 modes.

    Returns
    -------
    func_value : float or 2darray
                 The function at the requested
                 angular coordinates.
    """
    message("Ang contracting", message_verbosity=3)

    # try:
    #    sh = theta.shape
    #    func_value = np.zeros(sh, dtype=np.complex128)
    # except Exception as ex:
    #    print(ex)
    #    func_value = 0

    ell_max, all_ells = GetEllMaxFromModesDict(modes_Clm)

    func_values = np.array(
        [
            modes_Clm[f"l{ell}"][f"m{emm}"]
            * Yslm_vec(spin_weight=0, ell=ell, emm=emm, theta_grid=theta, phi_grid=phi)
            for ell in range(ell_max + 1)
            for emm in range(-ell, ell + 1)
        ]
    )
    # func_value += emm_entry*Yslm_vec(spin_weight=0, ell=ell, emm=emm, theta_grid=theta, phi_grid=phi)
    message("Func values shape", func_values.shape, message_verbosity=4)
    func_value = np.sum(func_values, axis=(0))

    message("Done AngCont", message_verbosity=3)
    return func_value


def jacobian_spherical_polar_to_cartesian(spherical_polar_coords):
    """Compute the jacobian from grid frame spherical polar coordinate system
    to cartesian coordinate system"""

    radius, theta, phi = grid_frame_sphp_coords

    St = np.sin(theta)
    Ct = np.cos(theta)

    Sp = np.sin(phi)

    Cp = np.cos(phi)

    zro = np.zeros(St.shape)

    Jac = np.array(
        [
            [St * Cp, radius * Ct * Cp, -radius * St * Sp],
            [St * Sp, radius * Ct * Sp, radius * St * Cp],
            [Ct, -radius * St, zro],
        ]
    )

    return Jac


def jacobian_cartesian_to_spherical_polar(cartesian_coords):
    """Inverse of `jacobian_grid_frame_spherical_polar_to_grid_frame_cartesian`"""

    spherical_polar_coords = ToSphericalPolar(cartesian_coords)

    Jac_inv = jacobian_spherical_polar_to_cartesian(spherical_polar_coords)

    Jac = np.linalg.inv(Jac_inv)

    return Jac


def compute_displacement_AB(coords_aha_center, coords_ahb_center):
    """Compute the displacement vector from AhB to AhA"""

    hca_x, hca_y, hca_z = coords_aha_center
    hcb_x, hcb_y, hcb_z = coords_ahb_center
    hab_x = hca_x - hcb_x
    hab_y = hca_y - hcb_y
    hab_z = hca_z - hcb_z

    return [hab_x, hab_y, hab_z]


def compute_x_grid_frame_angle(displacement_vector_AB):
    """Compute the angle that the axis joining the two
    black holes subtends with the global X axis given
    the displacement vector AB"""

    hab_x, hab_y, _ = displacement_vector_AB
    phiX = np.unwrap(np.arctan2(hab_y, hab_x))

    return phiX


def jacobian_grid_frame_spherical_polar_to_inertial_frame_cartesian(
    grid_frame_spherical_polar_coords, phiX
):

    radius, theta, phi = grid_frame_spherical_polar_coords
    phi_inertial = phi + phiX
    anc_spherical_polar_coords = [radius, theta, phi_inertial]

    return jacobian_spherical_polar_to_cartesian(anc_spherical_polar_coords)


def transform_coordinate_system(t_ref, 
                                waveform_times, 
                                waveform_modes, 
                                horizon_times,
                                xA, 
                                xB,
                                massA,
                                massB,
                                chiA, 
                                chiB, 
                                chiC_f,
                                velC_f,
                                flipped=False
                                ):
    """Transform the spins and the waveform modes of the binary from the 
    ID coordinates to the coordinate system at t_ref
    """

    # Find the index of the reference time on the horizon
    horizon_t_ref_idx = np.argmin(abs(horizon_times - t_ref))
    Lhat = compute_angular_momentum_direction(horizon_t_ref_idx, 
                                              xA, 
                                              xB,
                                              massA,
                                              massB,
                                             )

    Omegahat =  compute_rotation_plane_normal(horizon_t_ref_idx, 
                                              xA, 
                                              xB,
                                             )

    #np.testing.assert_array_almost_equal(Lhat, Omagahat, 8, "The rotation plane direction is different from Angular momentum direction")
    print("Lhat and omegahat", Lhat, Omegahat)

    xA_rot_z, xB_rot_z, chiA_rot_z, chiB_rot_z, chiC_f_rot_z, velC_f_rot_z, waveform_modes_rot_z = \
                                align_parameters_along_z(waveform_times,
                                                         xA, 
                                                         xB,
                                                         chiA, 
                                                         chiB, 
                                                         chiC_f,
                                                         velC_f, 
                                                         waveform_modes,
                                                         Lhat
                                                         )
    print(chiC_f_rot_z)
    xA_rot_xyz, xB_rot_xyz, chiA_rot_xyz, chiB_rot_xyz, chiC_f_rot_xyz, velC_f_rot_xyz, waveform_modes_rot_xyz =\
                            align_parameters_along_xy(horizon_t_ref_idx, 
                                                      waveform_times, 
                                                      waveform_modes_rot_z, 
                                                      xA_rot_z, 
                                                      xB_rot_z,
                                                      chiA_rot_z, 
                                                      chiB_rot_z, 
                                                      chiC_f_rot_z,
                                                      velC_f_rot_z,)

    print(chiC_f_rot_xyz)
    return xA_rot_xyz, xB_rot_xyz, chiA_rot_xyz, chiB_rot_xyz, chiC_f_rot_xyz[0], velC_f_rot_xyz[0], waveform_modes_rot_xyz


def compute_angular_momentum_direction(horizon_t_ref_idx, 
                                       xA, 
                                       xB,
                                       massA,
                                       massB,):
    

    # Compute angular momentum (ignore dt for direction)
    dxA = np.diff(xA, axis=0)[horizon_t_ref_idx]
    dxB = np.diff(xB, axis=0)[horizon_t_ref_idx]
    pAdt = massA*dxA
    pBdt = massB*dxB
    lAdt = np.cross(xA[horizon_t_ref_idx], pAdt)
    lBdt = np.cross(xB[horizon_t_ref_idx], pBdt)
    Ldt = lAdt + lBdt
    Lhat = Ldt/np.sqrt((np.dot(Ldt, Ldt)))
    return Lhat


def compute_rotation_plane_normal(horizon_t_ref_idx, 
                                       xA, 
                                       xB,
                                    ):
    

    # Compute angular momentum (ignore dt for direction)
    dxA = np.diff(xA, axis=0)[horizon_t_ref_idx]
    dxB = np.diff(xB, axis=0)[horizon_t_ref_idx]
 
    omegaAdt = np.cross(xA[horizon_t_ref_idx], dxA)
    omegaBdt = np.cross(xB[horizon_t_ref_idx], dxB)

    #Ldt = lAdt + lBdt
    omegadt = omegaAdt

    omegaAhat = omegaAdt/np.sqrt((np.dot(omegaAdt, omegaAdt)))
    omegaBhat = omegaBdt/np.sqrt((np.dot(omegaBdt, omegaBdt)))
    
    print("Omegahats", omegaAhat, omegaBhat)

    omegahat = omegadt/np.sqrt((np.dot(omegadt, omegadt)))

    return omegahat


def align_parameters_along_z(waveform_times,
                             xA, 
                             xB,
                             chiA, 
                             chiB, 
                             chiC_f,
                             velC_f, 
                             waveform_modes,
                             Lhat
                            ):

    n_hor_times, _ = xA.shape
    n_wfm_times = len(waveform_times)
    # Align the z-direction
    q0_z = rotations.alignVec_quat(Lhat)
    q0_vec_z = np.array([q0_z]*n_hor_times).T
    xA_rot_z = rotations.transformTimeDependentVector(q0_vec_z, 
                                                      xA.T, 
                                                      inverse=1).T
    xB_rot_z = rotations.transformTimeDependentVector(q0_vec_z, 
                                                      xB.T, 
                                                      inverse=1).T
    chiA_rot_z = rotations.transformTimeDependentVector(q0_vec_z, 
                                                        chiA.T, 
                                                        inverse=1).T
    chiB_rot_z = rotations.transformTimeDependentVector(q0_vec_z, 
                                                        chiB.T, 
                                                        inverse=1).T

    if np.shape(chiC_f) != (3,):
        raise ValueError('Expected a single spin triple for chiC_f')
    
    chiC_f_rot_z = rotations.transformTimeDependentVector(np.array([q0_z]).T, 
                                                        np.array([chiC_f]).T, 
                                                        inverse=1).T
    
    if np.shape(velC_f) != (3,):
        raise Exception('Expected a single spin triple for velC_f_coord')
    
    velC_f_rot_z = rotations.transformTimeDependentVector(np.array([q0_z]).T,
                                                        np.array([velC_f]).T, 
                                                        inverse=1).T
    q0_wfm_z = q0_vec_z = np.array([q0_z]*n_wfm_times).T
    waveform_modes_rot_z = rotations.transformWaveform(waveform_times, 
                                                       q0_wfm_z, 
                                                       waveform_modes, 
                                                       inverse=1)

    print(chiC_f_rot_z)
    return xA_rot_z, xB_rot_z, chiA_rot_z, chiB_rot_z, chiC_f_rot_z, velC_f_rot_z, waveform_modes_rot_z


def align_parameters_along_xy(horizon_t_ref_idx, 
                             waveform_times, 
                             waveform_modes_rot_z, 
                             xA_rot_z, 
                             xB_rot_z,
                             chiA_rot_z, 
                             chiB_rot_z, 
                             chiC_f_rot_z,
                             velC_f_rot_z,
                             ):


    # Align in the x-y plane
    phi_A = np.angle(xA_rot_z[horizon_t_ref_idx,0] + 1.j*xA_rot_z[horizon_t_ref_idx,1])
    phi_B = np.angle(-xB_rot_z[horizon_t_ref_idx,0] - 1.j*xB_rot_z[horizon_t_ref_idx,1])

    # If this fails, try aligning at earlier times
    dphase_ang = abs(np.angle(np.exp(1.j*(phi_A - phi_B))))
    if dphase_ang > 0.15:
        print(horizon_t_ref_idx)
        print(min(waveform_times), max(waveform_times))
        raise ValueError(f"Got different x-y rotations from the black holes! err={dphase_ang}")
    
    n_hor_times, _ = xA_rot_z.shape
    n_wfm_times = len(waveform_times)

    q1_xy = rotations.zRotationQuat(phi_A)
    q1_xy_vec = np.array([q1_xy]*n_hor_times).T

    xA_rot_xyz = rotations.transformTimeDependentVector(q1_xy_vec, 
                                                        xA_rot_z.T, 
                                                        inverse=1).T
    xB_rot_xyz = rotations.transformTimeDependentVector(q1_xy_vec, 
                                                        xB_rot_z.T, 
                                                        inverse=1).T
    chiA_rot_xyz = rotations.transformTimeDependentVector(q1_xy_vec, 
                                                          chiA_rot_z.T, 
                                                          inverse=1).T
    chiB_rot_xyz = rotations.transformTimeDependentVector(q1_xy_vec, 
                                                          chiB_rot_z.T, 
                                                          inverse=1).T
    chiC_f_rot_xyz = rotations.transformTimeDependentVector(np.array([q1_xy]).T, 
                                                          chiC_f_rot_z.T,
                                                          inverse=1).T
    velC_f_rot_xyz = rotations.transformTimeDependentVector(np.array([q1_xy]).T,
                                                          velC_f_rot_z.T, 
                                                          inverse=1).T
    waveform_modes_rot_xyz = rotations.transformWaveform(waveform_times, 
                                                         np.array([q1_xy]*n_wfm_times).T, 
                                                         waveform_modes_rot_z, 
                                                         inverse=1)

    print(chiC_f_rot_xyz)

    return xA_rot_xyz, xB_rot_xyz, chiA_rot_xyz, chiB_rot_xyz, chiC_f_rot_xyz, velC_f_rot_xyz, waveform_modes_rot_xyz 

def plot_trajectories():

    import matplotlib.pyplot as plt
    # Two subplots
    fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
    # Plot x-y coords of BhA
    ax = axarr[0]
    ax.plot(xA_aln.T[0], xA_aln.T[1])
    ax.plot(xA_aln[iAlign][0], xA_aln[iAlign][1], 'ro')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_xlabel('xA')
    ax.set_ylabel('yA')
    # Plot x-y coords of BhB
    ax = axarr[1]
    ax.plot(xB_aln.T[0], xB_aln.T[1])
    ax.plot(xB_aln[iAlign][0], xB_aln[iAlign][1], 'ro')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_xlabel('xB')
    ax.set_ylabel('yB')
    plt.savefig('xA_aln.png', bbox_inches='tight')
    plt.close()
    exit()


def quaternionic_coordinate_transform_sanity_check(t_ref, 
                                waveform_times, 
                                waveform_modes, 
                                horizon_times,
                                xA, 
                                xB, 
                                massA,
                                massB,
                                chiA, 
                                chiB, 
                                chiC_f,
                                velC_f,
                                flipped=False):
    # Some sanity checks
    if abs(horizon_times[0]) > 1.:
        raise ValueError(f"Expected t_horizon[0]=0, got {horizon_times[0]}")

    max_horizon_dt =  max(np.diff(horizon_times))
    if max_horizon_dt> 5.:
        raise ValueError(f"Largest time step is {max_horizon_dt} - missing horizon data?")
    max_waveform_dt = max(np.diff(waveform_times))
    if max_waveform_dt > 5.:
        raise ValueError(f"Largest time step is {max_waveform_dt} - missing waveform data?")
    if abs(xA[0, 1]) + abs(xA[0,2]) > 0.5:
        raise ValueError(f"Expected xA to start roughly on the x-axis: {xA[0]}")
    if abs(xB[0,1]) + abs(xB[0,2]) > 0.5:
        raise ValueError(f"Expected xB to start roughly on the x-axis: {xB[0]}")
    if (not flipped) and (xA[0,0] < 0.5 or xB[0,0] > -1.):
        raise ValueError("Expected xA to be on the +ive x-axis, xB on -ive.")
    if flipped and (xA[0,0] > -1. or xB[0,0] < 1.):
        raise ValueError("Expected xA to be on the -ive x-axis for flipped")
