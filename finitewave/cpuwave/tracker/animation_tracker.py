from pathlib import Path
import shutil
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class AnimationTracker(Tracker):
    """
    A class to track and save frames of a cardiac tissue model simulation
    for animation purposes.

    This tracker periodically saves the state of a specified target array from
    the model to disk as NumPy files, which can later be used to create
    animations.

    Saves frames as .npy snapshots, then can build an animation via:
    - Animation2DBuilder for 2D fields
    - Animation3DBuilder for 3D fields
    """

    def __init__(self):
        super().__init__()
        self.dir_name = "animation"
        self.variable_name = "u"
        self.frame_type = "float64"
        self._frame_counter = 0
        self.overwrite = True

        # Optional: allow forcing dimension (if None: auto from model.u.ndim)
        self.ndim = None

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and sets up
        directories for saving frames.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        self._frame_counter = 0 # Reset frame counter

        dir_path = Path(self.path, self.dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.overwrite:
            for file in dir_path.glob("*.npy"):
                file.unlink()

        # cache ndim for write()
        if self.ndim is None:
            self._ndim = getattr(self.model, "u").ndim
        else:
            self._ndim = int(self.ndim)

    def _track(self):
        # grab target field
        arr = self.model.__dict__[self.variable_name]
        frame = arr.copy()

        # set outside tissue to nan (works for both 2D/3D)
        mesh = self.model.cardiac_tissue.mesh
        frame[mesh != 1] = np.nan

        dir_path = Path(self.path, self.dir_name)
        np.save(
            dir_path.joinpath(str(self._frame_counter)).with_suffix(".npy"),
            frame.astype(self.frame_type),
        )
        self._frame_counter += 1

    def write(self, path=None, animation_name=None, clear=False, prog_bar=True, **kwargs):
        """
        Build an animation from saved frames.

        Parameters (common)
        -------------------
        path : str|Path, optional
            Where to save the animation. Defaults to tracker path.
        animation_name : str, optional
            Output filename (without extension). Defaults to dir_name.
        clear : bool, optional
            Remove snapshots after building animation.
        prog_bar : bool, optional
            Show progress bar.
        **kwargs :
            Passed to the underlying builder, depending on 2D/3D.

        2D kwargs (commonly used)
        -------------------------
        shape_scale=1, fps=12, cmap="coolwarm", clim=[0,1]

        3D kwargs (commonly used)
        -------------------------
        cmap="viridis", clim=[0,1], scalar_bar=False, format="mp4", plus builder-specific kwargs
        """
        path_load = Path(self.path, self.dir_name)
        path_save = Path(self.path) if path is None else Path(path)
        name = self.dir_name if animation_name is None else animation_name

        mesh = self.model.cardiac_tissue.mesh
        ndim = self._ndim

        # Lazy imports to keep tracker usable without tools extras:
        try:
            from finitewave.tools.animation_2d_builder import Animation2DBuilder
        except ImportError as e:
            raise ImportError(
                "Building animations requires optional dependencies.\n\n"
                "Install one of the following:\n"
                "  • Install all tools:\n"
                "      pip install \"finitewave[tools]\"\n"
                "  • Install only what you need for 2D MP4:\n"
                "      pip install natsort ffmpeg-python\n"
                "    plus system FFmpeg (binary):\n"
                "      Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "      macOS: brew install ffmpeg\n"
                "      Conda: conda install -c conda-forge ffmpeg\n"
                "      Windows: winget install Gyan.FFmpeg\n"
            ) from e

        # Import 3D builder only if needed (so 2D users don't need pyvista):
        Animation3DBuilder = None
        if ndim == 3:
            try:
                from finitewave.tools.animation_3d_builder import Animation3DBuilder
            except ImportError as e:
                raise ImportError(
                    "Building 3D animations requires optional dependencies.\n\n"
                    "Install one of the following:\n"
                    "  • Install all tools:\n"
                    "      pip install \"finitewave[tools]\"\n"
                    "  • Install only PyVista (3D rendering):\n"
                    "      pip install pyvista\n"
                    "    (MP4 export may also require system FFmpeg depending on your setup.)\n"
                ) from e

        if ndim == 2:
            # Defaults for 2D
            shape_scale = kwargs.pop("shape_scale", 1)
            fps = kwargs.pop("fps", 12)
            cmap = kwargs.pop("cmap", "coolwarm")
            clim = kwargs.pop("clim", [0, 1])

            # 2D builder expects boolean mask (outside tissue)
            mask = (mesh != 1)

            builder = Animation2DBuilder()
            builder.write(
                path_load,
                path_save=path_save,
                animation_name=name,
                mask=mask,
                shape_scale=shape_scale,
                fps=fps,
                clim=clim,
                shape=mesh.shape,
                cmap=cmap,
                prog_bar=prog_bar,
                **kwargs,
            )

        elif ndim == 3:
            # Defaults for 3D
            cmap = kwargs.pop("cmap", "viridis")
            clim = kwargs.pop("clim", [0, 1])
            scalar_bar = kwargs.pop("scalar_bar", False)
            format_ = kwargs.pop("format", "mp4")

            mask = mesh  # or (mesh != 1) depending on Animation3DBuilder API

            builder = Animation3DBuilder()
            builder.write(
                path_load,
                path_save=path_save,
                animation_name=name,
                mask=mask,
                scalar_name=self.variable_name,
                clim=clim,
                cmap=cmap,
                scalar_bar=scalar_bar,
                format=format_,
                prog_bar=prog_bar,
                **kwargs,
            )

        else:
            raise ValueError(f"Unsupported ndim={ndim} for animation. Expected 2 or 3.")

        if clear:
            shutil.rmtree(path_load)