"""
Running the Fentom-Karma Model in 3D
======================================

Overview:
---------
This example demonstrates how to run a basic 3D simulation of the 
Fentom-Karma model using the Finitewave framework. 

Simulation Setup:
-----------------
- Tissue Grid: A 100×5×3 cardiac tissue domain.
- Stimulation:
  - A square side stimulus is applied at t = 0.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 500

Execution:
----------
1. Create a 3D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Fentom-Karma model.
4. Visualize the transmembrane potential.

"""

import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue:
n = 100
m = 5
k = 3
tissue = fw.CardiacTissue3D([n, m, k])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, 5, 0, m, 0, k))

# create model object and set up parameters:
fentom_karma = fw.FentonKarma3D()
fentom_karma.dt = 0.01
fentom_karma.dr = 0.25
fentom_karma.t_max = 500
# add the tissue and the stim parameters to the model object:
fentom_karma.cardiac_tissue = tissue
fentom_karma.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3, 1]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
fentom_karma.tracker_sequence = tracker_sequence

# run the model:
fentom_karma.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * fentom_karma.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3_1")
plt.legend(title='Fentom-Karma')
plt.xlabel('Time (ms)')
plt.title('Action Potential')
plt.grid()
plt.show()