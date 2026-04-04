"""
Parameter Regions in the Aliev–Panfilov Model (2D)
==================================================

Overview:
---------
This example demonstrates how to assign spatially varying model parameters
in the Aliev-Panfilov model using the Finitewave framework.

The tissue is divided into two horizontal regions with different values
of the excitability parameter `a`. A planar wave is initiated from the
bottom boundary, and the resulting activation dynamics are compared
between regions.

Simulation Setup:
-----------------
- Tissue Grid: A 100×100 2D cardiac tissue domain.
- Parameter Regions:
  - Lower half:  a = 0.1
  - Upper half:  a = 0.2
- Stimulation:
  - A planar stimulus is applied along the bottom boundary at t = 0.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 75

Expected Outcome:
-----------------
- The activation time map reveals different propagation dynamics
  in the two parameter regions.
- The region with higher `a` exhibits reduced excitability and
  altered conduction behavior.

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Define a spatial map of parameter `a`.
3. Apply a planar stimulus along the bottom boundary.
4. Run the Aliev–Panfilov model.
5. Visualize the activation time map.
"""


import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

# -----------------------------
# 1) Tissue
# -----------------------------
nx, ny = 100, 100
tissue = fw.CardiacTissue([nx, ny])

# -----------------------------
# 2) Stimulation (planar wave from the bottom)
# -----------------------------
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(
    fw.StimVoltageCoord(
        time=0,
        volt_value=1,
        x1=1, x2=4,  
        y1=1, y2=ny-1
    )
)

# -----------------------------
# 3) Model
# -----------------------------
model = fw.AlievPanfilov()
model.dt = 0.01
model.dr = 0.25
model.t_max = 75

model.cardiac_tissue = tissue
model.stim_sequence = stim_sequence

# -----------------------------
# 4) Parameter regions: a(x,y)
# -----------------------------
a_left = 0.1
a_right = 0.2

a_map = np.full((nx, ny), a_left, dtype=float)
a_map[:, ny//2:] = a_right  # right half has higher 'a'

# Set the spatially varying parameter map in the model
model.a = a_map

# -----------------------------
# 5) Trackers
# -----------------------------
trackers = fw.TrackerSequence()

act_time_tracker = fw.ActivationTimeTracker()
act_time_tracker.threshold = 0.5
act_time_tracker.step = 1
trackers.add_tracker(act_time_tracker)

model.tracker_sequence = trackers

model.run()

# -----------------------------
# 7) Plot activation time map
# -----------------------------

plt.figure()
plt.imshow(act_time_tracker.output, origin="lower", aspect="auto", cmap="coolwarm")
plt.axvline(nx//2 - 0.5, linestyle="--")  # region boundary
plt.title("activation time map")
plt.colorbar()
plt.tight_layout()
plt.show()
