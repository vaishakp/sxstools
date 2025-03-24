import numpy as np
from sxstools.transforms import transform_coordinate_system
from sxstools.transforms import rotations
from scri import WaveformModes
from scipy.interpolate import InterpolatedUnivariateSpline

class CoordinateTransform:
    ''' Transforms the coordinate system of a SpEC simulations from 
    that of ID to the one at a chosen t_ref.
    
    Transfrorms the spins, kick velocities and the WaveformModes accordingly.

    There are some choices one can make, viz.
    1. Specify an SXS WaveformModes obj as waveform_modes or just the modes and its time axis
    2. The choice of z axis to compute and use: Orbital angular momentum or normal to rotation plane
    3. If Omega_hat is chosen, one can spcify which object's angular momentum to use.
    4. Choose fine alignment (by interpolation) or a rough one (by the closest time stamp)

    The last three have been retained for legacy comparisons with surragate_modelling implementation.

    '''

    def __init__(self,
                 t_ref,
                 waveform_modes,
                 massA,
                 massB,
                 xA,
                 xB,
                 chiA,
                 chiB,
                 chiC_final,
                 v_kick,
                 normal_direction='Lhat',
                 Omegahat_choice='AhA',
                 waveform_times=None,
                 method='fine',
                 additional_vectors_to_transform=None,
                 additional_vector_timeseries_to_transform=None,
                 return_wfm=True
                 ):
        
        self.t_ref = t_ref
        #if isinstance(waveform_modes, WaveformModes):
        if "WaveformModes" in str(type(waveform_modes)):
            self.waveform_modes = waveform_modes
            self.waveform_modes_data = waveform_modes.data.T
            self.waveform_times = waveform_modes.t

        elif isinstance(waveform_modes, np.ndarray):
            self.waveform_modes_data = waveform_modes
            self.waveform_modes = None
            if (np.array(waveform_times) == np.array(None)).any():
                raise KeyError("Please supply waveform time axis")
            else:
                self.waveform_times = waveform_times

        else:
            raise TypeError("Unknown waveform modes obj specified."
                            "Please supply an SXS WaveformModes obj or"
                             "an ndarray with modes along the first axis")
        
        self.massA = massA
        self.massB = massB
        self.xA = xA[:, 1:]
        self.xB = xB[:, 1:]
        self.chiA = chiA[:, 1:]
        self.chiB = chiB[:, 1:]
        self.chiC_final = chiC_final
        self.v_kick = v_kick
        self.horizon_times = xA[:, 0]
        self.t_ref_idx_horizon = np.argmin(abs(self.horizon_times - self.t_ref))
        self.n_wfm_times = len(self.waveform_times)
        self.n_hor_times = len(self.horizon_times)
        self.normal_direction=normal_direction
        self.Omegahat_choice = Omegahat_choice
        self.method = method
        self.additional_vectors_to_transform = additional_vectors_to_transform
        self.additional_vector_timeseries_to_transform = additional_vector_timeseries_to_transform
        self.return_wfm = return_wfm

        if self.method=='fine':
            self.eval = self.eval_rough
            self.eval_derivative = self.eval_derivative_fine
        else:
            self.eval = self.eval_rough
            self.eval_derivative = self.eval_derivative_rough

        self.vector_timeseries_to_transform = {'xA' : self.xA,
                                               'xB' : self.xB,
                                               'chiA': self.chiA, 
                                               'chiB': self.chiB
                                               }
        self.vectors_to_transform = {'chiC_final' : self.chiC_final,
                                     'v_kick' : self.v_kick
                                    }

        if self.additional_vectors_to_transform is not None:
            for item in self.vectors_to_transform:
                if item in self.additional_vectors_to_transform.keys():
                    raise ValueError(f"Cannot add additional item {item} to list as it is a default item")
            self.vectors_to_transform.update(self.additional_vectors_to_transform)


        if self.additional_vector_timeseries_to_transform is not None:
            for item in self.vector_timeseries_to_transform:
                if item in self.additional_vector_timeseries_to_transform.keys():
                    raise ValueError(f"Cannot add additional item {item} to list as it is a default item")
            self.vector_timeseries_to_transform.update(self.additional_vector_timeseries_to_transform)

        self.interpolants = {}
        self.construct_interpolants()

        # To hold the result
        self.transformed_quantities = {}
        self.ref_parameter_keys = ['chiA', 'chiB', 'chiC_final', 'v_kick']
        self.reference_parameters = {'massA' : massA,
                                     'massB' : massB}

    def construct_interpolants(self):
        ''' Construct interpolants for variables '''

        if self.method=='fine':
            for var_name, var_value in self.vector_timeseries_to_transform.items():
                self.interpolants.update({var_name : self.interpolate(self.horizon_times, var_value)})

    def construct_interpolants_rot_z(self):
        ''' Construct interpolants for timeseries variables '''
        if self.method=='fine':
            for var_name, var_value in self.transformed_quantities.items():
                if var_name in self.vector_timeseries_to_transform.keys():
                    if '_rot_z' in var_name:
                        self.interpolants.update({var_name : self.interpolate(self.horizon_times, var_value)})

    def get(self, var_name):
        if 'rot_' not in var_name:
            try:
                return self.vector_timeseries_to_transform[var_name]
            except KeyError:
                return self.vectors_to_transform[var_name]
        else:
            return self.transformed_quantities[var_name]
        
    def eval_fine(self, var_name, t_ref):
        _, ncols = self.vector_timeseries_to_transform[var_name].shape
        return np.array([self.interpolants[var_name][idx](t_ref) for idx in range(ncols)])
    
    def eval_rough(self, var_name, t_ref=None):
        try:
            return self.vector_timeseries_to_transform[var_name][self.t_ref_idx_horizon]
        except KeyError:
            return self.transformed_quantities[var_name][self.t_ref_idx_horizon]
        
    def eval_derivative_fine(self, var_name, t_ref):
        _, ncols = self.vector_timeseries_to_transform[var_name].shape
        return np.array([self.interpolants[var_name][idx].derivative()(t_ref) for idx in range(ncols)])

    def eval_derivative_rough(self, var_name, t_ref=None):
        return np.diff(self.vector_timeseries_to_transform[var_name], axis=0)[self.t_ref_idx_horizon]

    def interpolate(self, time, data):
        _, n_cols = data.shape

        interpolants = []
        for idx in range(n_cols):
            interpolants.append(InterpolatedUnivariateSpline(time, data[:, idx], k=5))
        
        return interpolants

    def compute_angular_momentum_direction(self):
        ''' Compute the unit vector in the direction of the 
        total angular momentum (orbital) of the BHs'''

        dxA = self.eval_derivative('xA', self.t_ref)
        dxB = self.eval_derivative('xB', self.t_ref)
        pAdt = self.massA*dxA
        pBdt = self.massB*dxB
        lAdt = np.cross(self.eval('xA', self.t_ref), pAdt)
        lBdt = np.cross(self.eval('xB', self.t_ref), pBdt)
        Ldt = lAdt + lBdt
        self.Lhat = Ldt/np.sqrt((np.dot(Ldt, Ldt)))
        
    def compute_rotation_plane_normal(self):
        ''' Compute the direction of the rotation plane of the
        BHs and return either one of them. '''

        # Compute rotation plane individually
        dxA = self.eval_derivative_rough('xA')
        dxB = self.eval_derivative_rough('xB')
        omegaAdt = np.cross(self.eval_rough('xA'), dxA)
        omegaBdt = np.cross(self.eval_rough('xB'), dxB)
        omegaAhat = omegaAdt/np.sqrt((np.dot(omegaAdt, omegaAdt)))
        omegaBhat = omegaBdt/np.sqrt((np.dot(omegaBdt, omegaBdt)))

        if self.Omegahat_choice=='AhA':
            omegadt = omegaAdt
        else:
            omegadt = omegaBdt

        print("Omegahats", omegaAhat, omegaBhat)
        self.Omegahat = omegadt/np.sqrt((np.dot(omegadt, omegadt)))


    def transform_one_vector_timeseries_along_z(self, var_name):
        ''' Transform the vector recognized by the key to the 
        coordinate system defined by the normal direction '''
        
        tvar_name = var_name+'_rot_z'
        result = rotations.transformTimeDependentVector(self.q0_vec_z, 
                                                        self.get(var_name).T,
                                                        inverse=1).T
        self.transformed_quantities.update({tvar_name : result})

    def transform_one_vector_along_z(self, var_name):
        ''' Transform the vector recognized by the key to the 
        coordinate system defined by the normal direction '''
        
        tvar_name = var_name+'_rot_z'
        result = rotations.transformTimeDependentVector(self.q0_z, 
                                                        np.array([self.get(var_name)]).T,
                                                        inverse=1).T
        self.transformed_quantities.update({tvar_name : result[0]})


    def transform_one_vector_along_xy(self, var_name):
        ''' Transform the vector recognized by the key to the 
        xy coordinate system defined at t_ref '''
        
        tvar_name = var_name.replace("rot_z", "rot_xyz")
        result = rotations.transformTimeDependentVector(self.q1_xy, 
                                                        np.array([self.get(var_name)]).T,
                                                        inverse=1).T
        self.transformed_quantities.update({tvar_name : result[0]})


    def transform_one_vector_timeseries_along_xy(self, var_name):
        ''' Transform the vector recognized by the key to the 
        xy coordinate system defined at t_ref '''
        
        tvar_name = var_name.replace("rot_z", "rot_xyz")
        result = rotations.transformTimeDependentVector(self.q1_vec_xy, 
                                                        self.get(var_name).T, 
                                                        inverse=1).T
        self.transformed_quantities.update({tvar_name : result})

    def align_along_z(self):
        ''' Align the quantities along the Lhat direction '''

        # Align the z-direction
        self.q0_z = rotations.alignVec_quat(self.Lhat)
        self.q0_vec_z = np.array([self.q0_z]*self.n_hor_times).T
        
        if np.shape(self.get('chiC_final')) != (3,):
            raise ValueError('Expected a single spin triple for chiC_final')
        
        if np.shape(self.get('v_kick')) != (3,):
            raise ValueError('Expected a single spin triple for v_kick')

        for var_name in self.vectors_to_transform.keys():
            self.transform_one_vector_along_z(var_name)

        for var_name in self.vector_timeseries_to_transform.keys():
            self.transform_one_vector_timeseries_along_z(var_name)

        q0_wfm_z = q0_vec_z = np.array([self.q0_z]*self.n_wfm_times).T
        self.waveform_modes_rot_z = rotations.transformWaveform(self.waveform_times, 
                                                        q0_wfm_z, 
                                                        self.waveform_modes_data, 
                                                        inverse=1)
        self.construct_interpolants_rot_z()


    def align_along_xy(self):
        ''' Align the coordinate system in the new xy directions as defined by
        the normal vector and the line joining the two objects '''

        self.compute_orbital_phase()
        self.q1_xy = rotations.zRotationQuat(self.phi_ref)
        self.q1_vec_xy = np.array([self.q1_xy]*self.n_hor_times).T

        for var_name in self.vectors_to_transform.keys():
            self.transform_one_vector_along_xy(var_name+"_rot_z")

        for var_name in self.vector_timeseries_to_transform.keys():
            self.transform_one_vector_timeseries_along_xy(var_name+"_rot_z")
        
        self.waveform_modes_rot_xyz = rotations.transformWaveform(self.waveform_times, 
                                                            np.array([self.q1_xy]*self.n_wfm_times).T, 
                                                            self.waveform_modes_rot_z,
                                                            inverse=1,
                                                            return_wfm=self.return_wfm)
        
        self.transformed_quantities.update({"waveform_modes_rot_xyz" : self.waveform_modes_rot_xyz})

    def compute_orbital_phase(self):
        ''' Compute the orbital phasing between the compact objects
        in the coordinate system aligned with Lhat at t_ref '''

        dX =  self.eval('xA_rot_z', self.t_ref)[0] -self.eval('xB_rot_z', self.t_ref)[0]
        dY =  self.eval('xA_rot_z', self.t_ref)[1] -self.eval('xB_rot_z', self.t_ref)[1]
        dX_ts =  self.get('xA_rot_z')[:, 0] -self.get('xB_rot_z')[:, 0]
        dY_ts =  self.get('xA_rot_z')[:, 1] -self.get('xB_rot_z')[:, 1]
        self.phi_ts = np.array(np.angle(dX_ts + 1j*dY_ts))
        #self.vector_timeseries_to_transform.update({'phiAB' : phi_ts})
        #print(phi_ts.shape)
        #self.interpolants.update({"phiAB" : self.interpolate(self.horizon_times, phi_ts)})
        self.phi_ref = np.angle(dX + 1j*dY)
        dr = np.array([dX, dY])
        nhat = dr/(np.sqrt(np.dot(dr, dr)))
        self.nhat = nhat
        print("nhat", self.nhat)

    def compute_orbital_phase_legacy(self):
        '''Compute the orbital phasing in the coordinate system aligned with Lhat at t_ref
        as the phase that one of the objects makes with the X axis? '''

        # Does this not retain the axis from ID frame?
        # Why not compute this using the relative orientation?
        phi_A = np.angle(self.eval_rough('xA_rot_z')[0] + 1.j*self.eval_rough('xA_rot_z')[1])
        phi_B = np.angle(-self.eval_rough('xB_rot_z')[0] - 1.j*self.eval_rough('xB_rot_z')[1])
        dphase_ang = abs(np.angle(np.exp(1.j*(phi_A - phi_B))))

        if dphase_ang > 0.15:
           print("Horiozon ref idx:", self.t_ref_idx_horizon)
           print("Waveform time extremes: min(self.waveform_times), max(self.waveform_times)")
           raise ValueError(f"Got different x-y rotations from the black holes! phase err={dphase_ang}")

        self.phi_ref = self.phi_A

    def transform(self):
        ''' Transform the parameters to the new frame at t_ref '''

        self.compute_angular_momentum_direction()
        self.compute_rotation_plane_normal()
        print(f"Lhat {self.Lhat}, Omegahat: {self.Omegahat}")

        if self.normal_direction=='Lhat':
            self.z_hat = self.Lhat
        elif self.normal_direction=='Omegahat':
            self.z_hat = self.Omegahat
        else:
            raise KeyError(f"Unknown normal direction {self.normal_direction}")
        
        self.align_along_z()
        self.align_along_xy()
        self.compute_reference_values()

    def compute_reference_values(self):
        
        for var_name in self.ref_parameter_keys:
            if var_name in self.vector_timeseries_to_transform.keys():
                self.interpolants.update({f'{var_name}_rot_xyz': self.interpolate(self.horizon_times, self.get(f'{var_name}_rot_xyz'))})
                self.reference_parameters.update({var_name : self.eval(f"{var_name}_rot_xyz", self.t_ref)})
            
            elif var_name in self.vectors_to_transform.keys():
                self.reference_parameters.update({var_name : self.get(f"{var_name}_rot_xyz")})


        self.omega_ref = InterpolatedUnivariateSpline(self.horizon_times, self.phi_ts, k=5).derivative()(self.t_ref)
        self.reference_parameters.update({'phi' : self.phi_ref})
        self.reference_parameters.update({'omega' : self.omega_ref})