# Changelog

## [0.9.0]

### Added

- Node-specific model parameters. You can now define model parameters as arrays with individual values for each mesh node. See the **parameter_regions.py** example.

- External model library support. Cardiac models are now maintained in separate repositories and used as dependencies by Finitewave.

- Observers. You can now attach observers that are evaluated during the simulation. See the **observers.py** example.

---

### Changed

- Removed **2D/3D** prefixes from class names.  
  Example: use **CardiacTissue** instead of **CardiacTissue2D** or **CardiacTissue3D**.  
  The framework automatically adjusts dimensionality based on the mesh.

- Kernel generation system. Finitewave now uses kernel generators to compose computational steps.  
  This enables node-specific parameters and observer integration.

- **VariablesTracker** replaces both **VariableTracker** and **MultiVariableTracker**.  
  Use **VariablesTracker** in the same way as the former **MultiVariableTracker**.

- Finitewave is now available via **pip**.

---

### Notes

- The old **2D/3D** class prefixes are still supported for backward compatibility.  
  Existing scripts remain compatible with this version.
