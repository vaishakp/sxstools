import numpy as np
import h5py
from waveformtools.waveformtools import message
from pathlib import Path
from sxstools.transforms import GetSphereRadialExtents, GetDomainRadii
from spectral.spherical.grids import GLGrid
from spectral.chebyshev.chebyshev import ChebyshevSpectral

class SXSDataLoader:


    def __init__(self,
                 run_dir,
                 name='SphereC0',
                 metric_dir='PsiKappa'):

        self.run_dir = Path(run_dir)
        self._data = {}
        self.name = name
        self.metric_dir = metric_dir

    @property
    def data(self):
        return self._data

    def load_grid_structure(self):
        filev = self.run_dir/f"{self.metric_dir}/Vars_{self.name}.h5"
        vars_dat = h5py.File(filev)
        self.n_radii, self.n_theta, self.n_phi = vars_dat['psi']['Step000000'].attrs['Extents']
        self.n_time = len(list(vars_dat['psi'].keys()))

        message(f"Ntime {self.n_time} \t N_radii {self.n_radii} N_theta {self.n_theta} N_phi {self.n_phi}")

        filed = self.run_dir/"GrDomain.input"
        
        radii_dict = GetDomainRadii(filed)

        message(f"Available radii data {radii_dict.keys()}")
        
        try:
            self.radial_collocation_points = radii[self.name]
            self.r_min, self.r_max = GetSphereRadialExtents(radii_dict, sub_domain=self.name)
        except Exception as excep:
            message(f" {excep}: \t Radial collocation points data not found for SubDomain {self.name}. Retreiving from SphereC0")
            
            self.radial_collocation_points = radii["SphereC"]
            self.r_min, self.r_max = GetSphereRadialExtents(radii_dict, sub_domain="SphereC0")

        assert len(self.radial_collocation_points) == self.n_radii, "Number of radial collocation points obtained from GrDomain.input does not match that from the data!"

    def get_angular_grid(self):

        self.Grid = GLGrid(L=self.n_theta-1)

    def get_radial_grid(self):
        
        self.radial_grid = ChebyshevSpectral(
            a=self.r_min, b=self.r_max, Nfuncs=self.n_radii
        )

    def get_four_metric_data(self, t_step):
        pass

    def load_Psi4_data(self):
        pass

