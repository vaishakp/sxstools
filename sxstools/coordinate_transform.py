import numpy as np
from sxstools.transforms import transform_coordinate_system
from sxstools.transforms import rotations
from scri import WaveformModes

class CoordinateTransform:
    ''' Transforms the coordinate system of a SpEC simulations from 
    that of ID to the one at a chosen t_ref.
    
    Transfrorms the spins, kick velocities and the WaveformModes accordingly.

    There are some choices one can make, viz.
    1. Specify an SXS WaveformModes obj as waveform_modes or just the modes and its time axis
    2. The choice of z axis to compute and use: Orbital angular momentum or normal to rotation plane
    3. If Omega_hat is chosen, one can spcify which object's angular momentum to use.

    The last two have been retained for legacy comparisons with surragate_modelling choices.
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
                 waveform_times=None
                 ):
        
        self.t_ref = t_ref
        if isinstance(waveform_modes, WaveformModes):
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
        self.t_ref_horizon = np.argmin(abs(self.horizon_times - self.t_ref))
        self.n_wfm_times = len(self.waveform_times)
        self.n_hor_times = len(self.horizon_times)
        self.normal_direction=normal_direction
        self.Omegahat_choice = Omegahat_choice

        # To hold the result
        self.transformed_quantities = {}


    def compute_angular_momentum_direction(self):
        ''' Compute the unit vector in the direction of the 
        total angular momentum (orbital) of the BHs'''

        # Compute angular momentum (ignore dt for direction)
        dxA = np.diff(self.xA, axis=0)[self.t_ref_horizon]
        dxB = np.diff(self.xB, axis=0)[self.t_ref_horizon]
        pAdt = self.massA*dxA
        pBdt = self.massB*dxB
        lAdt = np.cross(self.xA[self.t_ref_horizon], pAdt)
        lBdt = np.cross(self.xB[self.t_ref_horizon], pBdt)
        Ldt = lAdt + lBdt
        self.Lhat = Ldt/np.sqrt((np.dot(Ldt, Ldt)))


    def compute_rotation_plane_normal(self):
        ''' Compute the direction of the rotation plane of the
        BHs and return either one of them. '''

        # Compute rotation plane individually
        dxA = np.diff(self.xA, axis=0)[self.t_ref_horizon]
        dxB = np.diff(self.xB, axis=0)[self.t_ref_horizon]
        omegaAdt = np.cross(self.xA[self.t_ref_horizon], dxA)
        omegaBdt = np.cross(self.xB[self.t_ref_horizon], dxB)
        omegaAhat = omegaAdt/np.sqrt((np.dot(omegaAdt, omegaAdt)))
        omegaBhat = omegaBdt/np.sqrt((np.dot(omegaBdt, omegaBdt)))

        if self.Omegahat_choice=='AhA':
            omegadt = omegaAdt
        else:
            omegadt = omegaBdt

        print("Omegahats", omegaAhat, omegaBhat)
        self.Omegahat = omegadt/np.sqrt((np.dot(omegadt, omegadt)))

    def align_along_z(self):
        ''' Align the quantities along the Lhat direction '''

        # Align the z-direction
        q0_z = rotations.alignVec_quat(self.Lhat)
        q0_vec_z = np.array([q0_z]*self.n_hor_times).T
        self.xA_rot_z = rotations.transformTimeDependentVector(q0_vec_z, 
                                                        self.xA.T, 
                                                        inverse=1).T
        self.xB_rot_z = rotations.transformTimeDependentVector(q0_vec_z, 
                                                        self.xB.T, 
                                                        inverse=1).T
        self.chiA_rot_z = rotations.transformTimeDependentVector(q0_vec_z, 
                                                            self.chiA.T, 
                                                            inverse=1).T
        self.chiB_rot_z = rotations.transformTimeDependentVector(q0_vec_z, 
                                                            self.chiB.T, 
                                                            inverse=1).T

        if np.shape(self.chiC_final) != (3,):
            raise ValueError('Expected a single spin triple for chiC_final')
        
        self.chiC_final_rot_z = rotations.transformTimeDependentVector(np.array([q0_z]).T, 
                                                            np.array([self.chiC_final]).T, 
                                                            inverse=1).T
        
        if np.shape(self.v_kick) != (3,):
            raise ValueError('Expected a single spin triple for v_kick')
        
        self.v_kick_rot_z = rotations.transformTimeDependentVector(np.array([q0_z]).T,
                                                            np.array([self.v_kick]).T, 
                                                            inverse=1).T
        q0_wfm_z = q0_vec_z = np.array([q0_z]*self.n_wfm_times).T
        self.waveform_modes_rot_z = rotations.transformWaveform(self.waveform_times, 
                                                        q0_wfm_z, 
                                                        self.waveform_modes_data, 
                                                        inverse=1)


    def align_along_xy(self):
        ''' Align the coordinate system in the new xy directions as defined by
        the normal vector and the line joining the two objects '''

        self.compute_orbital_phase()
        q1_xy = rotations.zRotationQuat(self.phi_ref)
        q1_xy_vec = np.array([q1_xy]*self.n_hor_times).T

        self.xA_rot_xyz = rotations.transformTimeDependentVector(q1_xy_vec, 
                                                            self.xA_rot_z.T, 
                                                            inverse=1).T
        self.xB_rot_xyz = rotations.transformTimeDependentVector(q1_xy_vec, 
                                                            self.xB_rot_z.T, 
                                                            inverse=1).T
        self.chiA_rot_xyz = rotations.transformTimeDependentVector(q1_xy_vec, 
                                                            self.chiA_rot_z.T, 
                                                            inverse=1).T
        self.chiB_rot_xyz = rotations.transformTimeDependentVector(q1_xy_vec, 
                                                            self.chiB_rot_z.T, 
                                                            inverse=1).T
        self.chiC_final_rot_xyz = rotations.transformTimeDependentVector(np.array([q1_xy]).T, 
                                                            self.chiC_final_rot_z.T,
                                                            inverse=1).T
        self.v_kick_rot_xyz = rotations.transformTimeDependentVector(np.array([q1_xy]).T,
                                                            self.v_kick_rot_z.T, 
                                                            inverse=1).T
        self.waveform_modes_rot_xyz = rotations.transformWaveform(self.waveform_times, 
                                                            np.array([q1_xy]*self.n_wfm_times).T, 
                                                            self.waveform_modes_rot_z, 
                                                            inverse=1)

    def compute_orbital_phase(self):
        ''' Compute the orbital phasing between the compact objects
        in the coordinate system aligned with Lhat at t_ref '''

        dX =  self.xA_rot_z[self.t_ref_horizon,0] -self.xB_rot_z[self.t_ref_horizon,0] 
        dY =  self.xA_rot_z[self.t_ref_horizon,1] -self.xB_rot_z[self.t_ref_horizon,1]
        self.phi_ref = np.angle(dX + 1j*dY)
        
    def compute_orbital_phase_legacy(self):
        '''Compute the orbital phasing in the coordinate system aligned with Lhat at t_ref
        as the phase that one of the objects makes with the X axis? '''

        # Does this not retain the axis from ID frame?
        # Why not compute this using the relative orientation?
        phi_A = np.angle(self.xA_rot_z[self.t_ref_horizon,0] + 1.j*self.xA_rot_z[self.t_ref_horizon,1])
        phi_B = np.angle(-self.xB_rot_z[self.t_ref_horizon,0] - 1.j*self.xB_rot_z[self.t_ref_horizon,1])
        # If this fails, try aligning at earlier times
        dphase_ang = abs(np.angle(np.exp(1.j*(phi_A - phi_B))))

        if dphase_ang > 0.15:
           print("Horiozon ref idx:", self.t_ref_horizon)
           print("Waveform time extremes: min(self.waveform_times), max(self.waveform_times)")
           raise ValueError(f"Got different x-y rotations from the black holes! phase err={dphase_ang}")

        self.phi_ref = self.phi_A

    def transform(self):
        ''' Transform the parameters to the new frame aligned to t_ref '''

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
        keys = [ key for key in self.__dict__.keys() if 'xyz' in key]

        for key, val in self.__dict__.items():
            if 'xyz' in key:
                self.transformed_quantities.update({key : val})