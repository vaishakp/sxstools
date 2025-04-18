from waveformtools.waveformtools import message, get_val_at_t_ref
import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def get_dynamics_from_h5(full_path_to_h):
    ''' Get the dynamics of the system from the `Horizons.h5` file '''

    with h5py.File(full_path_to_h) as hhf:
        #message(sim_name)
        aha_dat = hhf["AhA.dir"]
        ahb_dat = hhf["AhB.dir"]
        ahc_dat = hhf["AhC.dir"]

        massA_arr = aha_dat["ChristodoulouMass.dat"][...][:, 1]
        massB_arr = ahb_dat["ChristodoulouMass.dat"][...][:, 1]
        massC_arr = ahc_dat["ChristodoulouMass.dat"][...][:, 1]
        massC_final = massC_arr[-1]

        #spinA_arr = aha_dat["DimensionfulInertialSpin.dat"][...]
        #spinB_arr = ahb_dat["DimensionfulInertialSpin.dat"][...]
        spinC_arr = ahc_dat["DimensionfulInertialSpin.dat"][...]
        spinC_final = spinC_arr[-1, 1:]

        chiA_arr = aha_dat['chiInertial.dat'][...][:, 1:]
        chiB_arr = ahb_dat['chiInertial.dat'][...][:, 1:]
        chiC_arr = ahc_dat['chiInertial.dat'][...][:, 1:]
        chiC_final = chiC_arr[-1]
        
        xA_arr = aha_dat['CoordCenterInertial.dat'][...][:, 1:]
        xB_arr = ahb_dat['CoordCenterInertial.dat'][...][:, 1:]
        xC_arr = ahc_dat['CoordCenterInertial.dat'][...][:, 1:]
        times   = aha_dat['CoordCenterInertial.dat'][...][:, 0]
        times_rd = ahc_dat['CoordCenterInertial.dat'][...][:, 0]
        v_kick_x = InterpolatedUnivariateSpline(times_rd, xC_arr[:, 0], k=3).derivative()(times[-1])
        v_kick_y = InterpolatedUnivariateSpline(times_rd, xC_arr[:, 1], k=3).derivative()(times[-1])
        v_kick_z = InterpolatedUnivariateSpline(times_rd, xC_arr[:, 2], k=3).derivative()(times[-1])

        v_kick = np.array([v_kick_x, v_kick_y, v_kick_z])

        #spinC_mag_final  = np.sqrt(np.dot(spinC_final, spinC_final))

        dynamics = {
                    "times" : times,
                    "xA" : xA_arr,
                    "xB" : xB_arr,
                    }
        
        omega = get_omega_from_dynamics(dynamics)

        dynamics.update( 
                            {       
                                    "omega": omega,
                                    "chiA" : chiA_arr,
                                    'chiB' : chiB_arr,
                                    'v_kick' : v_kick,
                                    'chiC_final' : chiC_final,
                                    'massA' : massA_arr,
                                    'massB' : massB_arr,
                                    'massC_final' : massC_final,
                            }
                        )
        
    return dynamics



def get_omega_from_dynamics(dynamics):
    ''' Compute the instantaneous orbital angular velocity
    from the dynamics dict '''

    time = dynamics["times"]
    xA = dynamics['xA']
    xB = dynamics['xB']

    dxAB = xA - xB
    vx = InterpolatedUnivariateSpline(time, dxAB[:, 0], k=3).derivative()(time)
    vy = InterpolatedUnivariateSpline(time, dxAB[:, 1], k=3).derivative()(time)
    vz = InterpolatedUnivariateSpline(time, dxAB[:, 2], k=3).derivative()(time)
    vAB = np.array([vx, vy, vz])
    dAB = np.sqrt(np.sum(dxAB*dxAB, axis=1))
    r2_omega_vec = np.cross(dxAB, vAB.T)
    omega = np.sqrt(np.sum(r2_omega_vec*r2_omega_vec, axis=1))/(dAB**2)

    return omega

def get_t_ref_from_dynamics_and_freq(dynamics, omega_ref=None, f_ref=None, Mtotal=None, t_junk=100):
    ''' Find the value of `t_ref` from the dynamics and supplied frequency '''
    if f_ref is not None:
        if Mtotal is not None:
            import lal
            omega_ref = np.pi*f_ref*Mtotal*lal.MTSUN_SI
            message("Omega_ref", omega_ref)
    else:
        raise ValueError("Please specify Mtotal")
            
    times = dynamics['times']
    omega = dynamics['omega']

    times_fine = np.linspace(t_junk, times[-1], 100000)
    omega_fine = InterpolatedUnivariateSpline(times, omega, k=3)(times_fine)
    t_ref = times_fine[np.argmin(abs(omega_fine-omega_ref))]

    return t_ref


def get_NR_ref_quantities_at_t_ref(t_ref, dynamics):
    ''' Fetch the reference quantities in NR frame at `t_ref` '''
    times     = dynamics["times"]
    massA_arr = dynamics["massA"]
    massB_arr = dynamics["massB"]

    chiA_arr = dynamics["chiA"]
    chiB_arr = dynamics["chiB"]

    massA_ref = get_val_at_t_ref(times, massA_arr, t_ref)
    massB_ref = get_val_at_t_ref(times, massB_arr, t_ref)

    chiAx_ref = get_val_at_t_ref(times, chiA_arr[:, 0], t_ref)
    chiAy_ref = get_val_at_t_ref(times, chiA_arr[:, 1], t_ref)
    chiAz_ref = get_val_at_t_ref(times, chiA_arr[:, 2], t_ref)

    chiA_ref_NR = np.array([chiAx_ref, chiAy_ref, chiAz_ref])

    chiBx_ref = get_val_at_t_ref(times, chiB_arr[:, 0], t_ref)
    chiBy_ref = get_val_at_t_ref(times, chiB_arr[:, 1], t_ref)
    chiBz_ref = get_val_at_t_ref(times, chiB_arr[:, 2], t_ref)

    chiB_ref_NR = np.array([chiBx_ref, chiBy_ref, chiBz_ref])

    ref_params = {"t_ref": t_ref,
                  "massA_ref" : massA_ref,
                  "massB_ref" : massB_ref,
                  "massC_final": dynamics["massC_final"],
                  "chiA_NR_ref": chiA_ref_NR,
                  "chiB_NR_ref": chiB_ref_NR,
                  "chiC_final": dynamics["chiC_final"],
                  "v_kick" : dynamics["v_kick"]
                 }
    
    return ref_params