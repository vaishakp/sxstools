from waveformtools.waveformtools import message, get_val_at_t_ref
import h5py
import numpy as np

def get_horizon_quantities_from_h5(full_path_to_h, t_ref):

    with h5py.File(full_path_to_h) as hhf:
        #message(sim_name)
        aha_dat = hhf["AhA.dir"]
        ahb_dat = hhf["AhB.dir"]
        ahc_dat = hhf["AhC.dir"]

        massA_arr = aha_dat["ChristodoulouMass.dat"][...]
        massB_arr = ahb_dat["ChristodoulouMass.dat"][...]
        massC_arr = ahc_dat["ChristodoulouMass.dat"][...]
        massC_final = massC_arr[-1, 1]

        spinA_arr = aha_dat["DimensionfulInertialSpin.dat"][...]
        spinB_arr = ahb_dat["DimensionfulInertialSpin.dat"][...]
        spinC_arr = ahc_dat["DimensionfulInertialSpin.dat"][...]
        spinC_final = spinC_arr[-1, 1:]

        chiA_arr = aha_dat['chiInertial.dat'][...]
        chiB_arr = ahb_dat['chiInertial.dat'][...]
        chiC_arr = ahc_dat['chiInertial.dat'][...]
        chiC_final = chiC_arr[-1, 1:]
        
        xA_arr = aha_dat['CoordCenterInertial.dat'][...]
        xB_arr = ahb_dat['CoordCenterInertial.dat'][...]
        xC_arr = ahc_dat['CoordCenterInertial.dat'][...]

        v_kick = np.diff(xC_arr, axis=0)[-1, 1:]
        spinC_mag_final  = np.sqrt(np.dot(spinC_final, spinC_final))

        massA_t_ref = get_val_at_t_ref(massA_arr[:, 0], massA_arr[:, 1], t_ref)
        massB_t_ref = get_val_at_t_ref(massB_arr[:, 0], massB_arr[:, 1], t_ref)

        parameters = {"xA" : xA_arr,
                                "xB" : xB_arr,
                                "chiA" : chiA_arr,
                                'chiB' : chiB_arr,
                                'v_kick' : v_kick,
                                'chiC_final' : chiC_final,
                                'massA' : massA_t_ref,
                                'massB' : massB_t_ref,
                                'massC' : massC_final,
                                }
        
    return parameters