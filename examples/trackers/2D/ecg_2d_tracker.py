
#
# Use the Period2DTracker to measure wave period (e.g spiral wave).
#

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
for model in [fw.AlievPanfilov2D]:
    aliev_panfilov = model()
    aliev_panfilov.dt = 0.01
    aliev_panfilov.dr = 0.25
    aliev_panfilov.t_max = 100

    # set up stimulation parameters:
    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimVoltageCoord2D(10, 1, 0, n, 0, 3))
    # stim_sequence.add_stim(StimVoltageCoord2D(31, 1, 0, 100, 0, n))

    tracker_sequence = fw.TrackerSequence()
    ecg_tracker = fw.ECG2DTracker()
    ecg_tracker.measure_points = np.array([[100, 100, 10]])
    tracker_sequence.add_tracker(ecg_tracker)

    # add the tissue and the stim parameters to the model object:
    aliev_panfilov.cardiac_tissue = tissue
    aliev_panfilov.stim_sequence = stim_sequence
    aliev_panfilov.tracker_sequence = tracker_sequence

    aliev_panfilov.run()

    plt.plot(ecg_tracker.ecg[0])
plt.show()
