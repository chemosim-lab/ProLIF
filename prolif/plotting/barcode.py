"""
Plot interactions as a barcode --- :mod:`prolif.plotting.barcode`
=================================================================

.. versionadded:: 2.0.0

.. autoclass:: Barcode
   :members:

"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from prolif.exceptions import RunRequiredError
from prolif.plotting.utils import IS_NOTEBOOK, separated_interaction_colors

if TYPE_CHECKING:
    from prolif.fingerprint import Fingerprint


class Barcode:
    """Creates a barcode plot of interactions.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame as obtained from :meth:`Fingerprint.to_dataframe()`

    Attributes
    ----------
    COLORS : dict
        Dictionnary of colors used in the plot for interactions.

    """

    COLORS: ClassVar[Dict[Optional[str], str]] = {
        None: "white",
        **separated_interaction_colors,
    }

    def __init__(self, df: pd.DataFrame) -> None:
        # mapping interaction type (HBond...etc.) to an arbitrary value which
        # corresponds to a color
        self.color_mapper = {
            interaction: value for value, interaction in enumerate(self.COLORS)
        }
        # reverse: map value to interaction type
        self.inv_color_mapper = {
            value: interaction for interaction, value in self.color_mapper.items()
        }
        # matplotlib colormap
        self.cmap = ListedColormap(list(self.COLORS.values()))

        # drop ligand level if single residue
        # else concatenate ligand with protein and drop ligand if peptide
        n_ligand_residues = len(np.unique(df.columns.get_level_values("ligand")))
        if n_ligand_residues == 1:
            df = df.droplevel("ligand", axis=1)
        else:
            df.columns = pd.MultiIndex.from_tuples(
                [(f"{items[0]}-{items[1]}", items[2]) for items in df.columns],
                names=["protein", "interaction"],
            )

        def _bit_to_color_value(s: pd.Series) -> pd.Series:
            """Replaces a bit value with it's corresponding color value"""
            interaction = s.name[-1]
            return s.apply(
                lambda v: self.color_mapper[interaction]
                if v
                else self.color_mapper[None]
            )

        self.df = df.astype(np.uint8).T.apply(_bit_to_color_value, axis=1)

    @classmethod
    def from_fingerprint(cls, fp: Fingerprint) -> Barcode:
        """Creates a barcode object from a fingerprint."""
        if not hasattr(fp, "ifp"):
            raise RunRequiredError(
                "Please run the fingerprint analysis before attempting to display results."
            )
        return cls(fp.to_dataframe())

    def display(
        self,
        figsize: Tuple[int, int] = (8, 10),
        dpi: int = 100,
        interactive: bool = IS_NOTEBOOK,
        n_frame_ticks: int = 10,
        residues_tick_location: Literal["top", "bottom"] = "top",
        xlabel: str = "Frame",
        subplots_kwargs: Optional[dict] = None,
        tight_layout_kwargs: Optional[dict] = None,
    ):
        """Generate and display the barcode plot.

        Parameters
        ----------
        figsize: Tuple[int, int] = (8, 10)
            Size of the matplotlib figure.
        dpi: int = 100
            DPI used for the matplotlib figure.
        interactive: bool
            Add hover interactivity to the plot (only relevant for notebooks). You may
            need to add ``%matplotlib notebook`` or ``%matplotlib ipympl`` for it to
            work as expected.
        n_frame_ticks: int = 10
            Number of ticks on the X axis. May use ±1 tick to have them evenly spaced.
        residues_tick_location: Literal["top", "bottom"] = "top"
            Whether the Y ticks appear at the top or at the bottom of the series of
            interactions of each residue.
        xlabel: str = "Frame"
            Label displayed for the X axis.
        subplots_kwargs: Optional[dict] = None
            Other parameters passed to :func:`matplotlib.pyplot.subplots`.
        tight_layout_kwargs: Optional[dict] = None
            Other parameters passed to :meth:`matplotlib.figure.Figure.tight_layout`.
        """
        if subplots_kwargs is None:
            subplots_kwargs = {}
        subplots_kwargs.setdefault("figsize", figsize)
        subplots_kwargs.setdefault("dpi", dpi)

        if tight_layout_kwargs is None:
            tight_layout_kwargs = {}
        tight_layout_kwargs.setdefault("pad", 1.2)

        # Plot as image
        fig, ax = plt.subplots(**subplots_kwargs)
        ax: plt.Axes
        im = ax.imshow(
            self.df.values,
            aspect="auto",
            interpolation="none",
            cmap=self.cmap,
            vmin=0,
            vmax=max(self.color_mapper.values()),
        )

        # Frame ticks
        frames = self.df.columns
        max_ticks = len(frames) - 1
        # try to have evenly spaced ticks
        for effective_n_ticks in (n_frame_ticks, n_frame_ticks - 1, n_frame_ticks + 1):
            samples, step = np.linspace(0, max_ticks, effective_n_ticks, retstep=True)
            if step.is_integer():
                break
        else:
            samples = np.linspace(0, max_ticks, n_frame_ticks)
        indices = np.round(samples).astype(int)
        ax.xaxis.set_ticks(indices, frames[indices])
        ax.set_xlabel(xlabel)

        # Residues ticks
        n_items = len(self.df.index)
        residues = self.df.index.get_level_values("protein")
        interactions = self.df.index.get_level_values("interaction")
        if residues_tick_location == "top":
            indices = [
                i
                for i in range(n_items)
                if (i - 1 >= 0 and residues[i - 1] != residues[i]) or i == 0
            ]
        else:
            indices = [
                i
                for i in range(n_items)
                if (i + 1 < n_items and residues[i + 1] != residues[i])
                or i + 1 == n_items
            ]
        ax.yaxis.set_ticks(indices, residues[indices])

        # legend
        values: List[int] = np.unique(self.df.values).tolist()
        try:
            values.pop(values.index(0))  # remove None color
        except ValueError:
            # 0 not in values (e.g. plotting a single frame)
            pass
        legend_colors = {
            self.inv_color_mapper[value]: im.cmap(value) for value in values
        }
        patches = [
            Patch(color=color, label=interaction)
            for interaction, color in legend_colors.items()
        ]
        ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)

        # interactive
        if interactive:
            self._add_interaction_callback(
                fig,
                ax,
                im=im,
                frames=frames,
                residues=residues,
                interactions=interactions,
            )

        fig.tight_layout(**tight_layout_kwargs)
        return ax

    def _add_interaction_callback(self, fig, ax, *, im, frames, residues, interactions):
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(5, 5),
            textcoords="offset points",
            alpha=0.8,
            bbox={"boxstyle": "round", "facecolor": "w"},
            wrap=True,
        )
        annot.set_visible(False)

        def hover_callback(event):
            if (
                event.inaxes is ax
                and event.xdata is not None
                and event.ydata is not None
            ):
                x, y = round(event.xdata), round(event.ydata)
                if self.df.values[y, x]:
                    annot.xy = (x, y)
                    frame = frames[x]
                    interaction = interactions[y]
                    residue = residues[y]
                    annot.set_text(f"Frame {frame}: {residue}")
                    color = im.cmap(self.color_mapper[interaction])
                    annot.get_bbox_patch().set_facecolor(color)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover_callback)
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
