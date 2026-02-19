"""
Using Observers in Finitewave
====================================================

Overview:
---------
This example demonstrates how to use observers in the Finitewave framework to make specific calculations or 
track variables during a cardiac electrophysiology simulation. 
We will run a 2D simulation of the Luo-Rudy 1991 ventricular action potential model and use an observer to track the IK current 
at a specific location in the tissue.

Simulation Setup:
-----------------
- Tissue Grid: A 100×100 cardiac tissue domain.
- Stimulation:
  - A planar stimulus is applied along the top edge of the domain at t = 0 ms
    to initiate wavefront propagation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 500 ms

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up the Luo-Rudy 1991 model and define an observer to track the IK current at a specific location.
4. Run the simulation.
5. Visualize the tracked IK current over time.

"""

import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

n = 100
# create mesh
tissue = fw.CardiacTissue((n, n))

# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1, 0, 5, 0, n))

# create model object and set up parameters
luo_rudy = fw.LuoRudy91()
luo_rudy.dt = 0.01
luo_rudy.dr = 0.25
luo_rudy.t_max = 500

# define an observer to track the IK current at a specific location (e.g., at cell (20, 30))
# add ik_obs - a numpy array to store the tracked IK current values at each time step). 
# The name "ik_obs" is used in the observer expression to refer to this array, and the expression updates the array at the specified location and time steps.
luo_rudy.ik_obs = np.zeros(int(luo_rudy.t_max/luo_rudy.dt/100), dtype=luo_rudy.npfloat)
luo_rudy.observers.append ({
    "name": "ik_obs",
    "expr": "if i_ == 20 and j_ == 30 and step % 100 == 0: ik_obs[step//100] = ik",  # example observer to track IK current
})

# add the tissue and the stim parameters to the model object
luo_rudy.cardiac_tissue = tissue
luo_rudy.stim_sequence = stim_sequence

# run the model:
luo_rudy.run()

# plot the action potential
plt.figure()
time = np.arange(len(luo_rudy.ik_obs)) * luo_rudy.dt
plt.plot(time, luo_rudy.ik_obs, label="IK current at (20, 30)")
plt.legend(title='Luo-Rudy 1991')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('IK Current')
plt.grid()
plt.show()
