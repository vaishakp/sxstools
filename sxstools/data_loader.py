import numpy as np
import h5py
from waveformtools.waveformtools import message
from pathlib import Path
from sxstools.transforms import GetSphereRadialExtents, GetDomainRadii
from spectral.spherical.grids import GLGrid
from spectral.chebyshev.chebyshev import ChebyshevSpectral
import os
from collections import namedtuple


class SXSDataLoader:

    def __init__(self, run_dir, subdomain="SphereC0", metric_dir="PsiKappa"):

        self.run_dir = Path(run_dir)
        self._data = {}
        self.subdomain = subdomain
        self.metric_dir = metric_dir

    @property
    def data(self):
        return self._data

    def load_grid_structure(self):

        # files = os.listdir(self.run_dir/f"{self.metric_dir}")
        # filev_suffix = [item for item in files if self.subdomain in item]

        # assert len(filev_suffix)==1, f"Multiple {self.subdomain}.h5 files found! {filev_suffix}"

        filev = self.run_dir / f"{self.metric_dir}/DumpedMetricData_{self.subdomain}.h5"
        vars_dat = h5py.File(filev)
        self.n_radii, self.n_theta, self.n_phi = vars_dat["psi"]["Step000000"].attrs[
            "Extents"
        ]
        self.n_time = len(list(vars_dat["psi"].keys()))

        message(
            f"Ntime {self.n_time} \t N_radii {self.n_radii} N_theta {self.n_theta} N_phi {self.n_phi}"
        )

        filed = self.run_dir / "GrDomain.input"

        radii_dict = GetDomainRadii(filed)

        message(f"Available radii data {radii_dict.keys()}")

        # try:
        self.radial_collocation_points = radii_dict[self.subdomain[:-1]]
        self.r_min, self.r_max = GetSphereRadialExtents(
            radii_dict, sub_domain=self.subdomain
        )

        # except Exception as excep:
        #    message(excep, f"\t Radial collocation points data not found for SubDomain {self.subdomain[:-1]}. Retreiving from SphereC0")

        #    self.radial_collocation_points = radii_dict["SphereC"]
        #    self.r_min, self.r_max = GetSphereRadialExtents(radii_dict, sub_domain="SphereC0")

        self.construct_angular_grid()
        self.construct_radial_grid()

    def construct_angular_grid(self):

        self.AngularGrid = GLGrid(L=self.n_theta - 1)
        self.theta_grid, self.phi_grid = self.AngularGrid.meshgrid

    def construct_radial_grid(self):

        self.RadialGrid = ChebyshevSpectral(
            a=self.r_min, b=self.r_max, Nfuncs=self.n_radii
        )

    def load_Psi4_data(self):
        pass

    def get_four_metric(self, t_step=0, component="tt"):

        fpath = self.run_dir / f"{self.metric_dir}/DumpedMetricData_{self.subdomain}.h5"
        dat_file = h5py.File(fpath)
        t_key = f"Step{str(t_step).zfill(6)}"

        reqd_data = (
            dat_file["psi"][t_key][component][...]
            .reshape(self.n_phi, self.n_theta, self.n_radii)
            .T
        )
        dat_file.close()

        return reqd_data

    def get_derivative_four_metric(self, t_step=0, component="ttt"):

        pass
