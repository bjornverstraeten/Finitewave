"""
Coordinates Stimulation
==================================

Overview:
---------
This example demonstrates how to apply a coordinate-based voltage stimulus in a 2D cardiac tissue

Stimulation Setup:
------------------
- The `StimVoltageCoord` class is used to define the stimulated region by its coordinates.
- A square region (10×10 nodes) at the top-left corner of the tissue is stimulated at t = 0.
- The voltage value is set to 1.0, which for the Aliev-Panfilov model corresponds to the peak excitation potential (resting = 0, peak = 1).

Simulation Parameters:
----------------------
- Model: Aliev-Panfilov 2D
- Grid size: 200 × 200
- Time step (dt): 0.01 
- Space step (dr): 0.25
- Total simulation time: 10 

Application:
This example is useful for learning how to define spatially localized voltage stimuli in 2D using coordinate-based methods. 
The `StimVoltageCoord` class is particularly useful for applying custom rectangular stimulation zones, making it ideal for 
simulating scenarios like localized pacing or focal arrhythmia initiation.
"""

import matplotlib.pyplot as plt

import finitewave as fw

# set up the tissue:
n = 200
tissue = fw.CardiacTissue([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
# stimulate the corner of the tissue with a square pulse (10 nodes on the side)
# of 1.0 V at t=0.
# coordinates are always form a reactangular (slab in 3D) area of stimulation.
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, 
                                             volt_value=1.0, 
                                             x1=0, x2=10, 
                                             y1=0, y2=10))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 25
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

# run the model:
aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(aliev_panfilov.u)
plt.colorbar()
plt.show()