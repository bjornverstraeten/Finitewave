# Finitewave examples

This directory contains a collection of **example scripts** demonstrating how to use the Finitewave framework for cardiac electrophysiology simulations.

The examples are organized into subdirectories by topic. They cover a range of use cases — from basic functionality to advanced simulation setups.

## Where to start?

If you are new to Finitewave, we recommend starting with:

1. `basics/` — to understand the core simulation pipeline.
2. `models/` — to explore individual electrophysiological models.
3. `stimulation/` and `trackers/` — to customize simulation workflows.

All examples are self-contained and can be modified to serve as templates for your own research workflows.

## Structure

### 📁 `basics/`

Examples of **basic framework usage** and common cardiac phenomena:

- How to initialize and run 2D and 3D simulations
- Visualization of wave propagation
- Modeling of typical phenomena such as **spiral waves/reentry**

### 📁 `models/`

**Minimal working examples** for each of the **electrophysiological models** implemented in Finitewave:

- Provide minimal working examples for each electrophysiological model implemented in Finitewave.

### 📁 `stimulation/`

Examples of different **stimulation protocols**:

- stimulation by current/voltage
- stimulation by coordinates, matrices
- making stimulation sequences

### 📁 `trackers/`

Examples of using **trackers** included in the framework:

- How to measure activation times, APD, electrograms (EGM), period maps, etc.
- How to record and analyze simulation results during runtime

### 📁 `fibrosis/`

Examples of **simulations in fibrotic tissue**:

- Preparing fibrosis maps
- Studying wave behavior in heterogeneous tissue

### 📁 `advanced/`

Examples of advanced Finitewave usage:

- Commands to control the simulation loop and define custom behavior.
- Observers to integrate additional checks or custom calculations into the computational kernel.

## How to run

You can run any example by executing it as a Python script:

```bash
python examples/<subdir>/<example_script.py>
