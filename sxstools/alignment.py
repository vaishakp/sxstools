from parallellib.parallel_multiprocessing import MultiprocessingClassTemplate



class QuadrupoleAlign(MultiprocessingClassTemplate):

    def __init__(self,
                 waveform_modes=None,
                 ):
        
        self.waveform_modes = waveform_modes
