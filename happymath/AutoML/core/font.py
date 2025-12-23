"""
Chinese font support mixin for AutoML visualizations.

Provides Chinese font rendering capabilities for matplotlib and plotly figures.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

# Import global Chinese font configuration
try:
    from ... import zh_font_available, zh_font_paths
except ImportError:
    zh_font_available = []
    zh_font_paths = {}


class ChineseFontMixin:
    """
    Chinese font support mixin.

    Provides Chinese font rendering capabilities for matplotlib and plotly charts.
    """

    def _get_chinese_font(self) -> str:
        """Get available Chinese font name."""
        if zh_font_available and isinstance(zh_font_available, list) and len(zh_font_available) > 0:
            return zh_font_available[0]
        return "DejaVu Sans"

    def _get_font_properties(self):
        """
        Get FontProperties object for Chinese display.
        Uses font paths detected in __init__.py directly.
        """
        from matplotlib.font_manager import FontProperties

        chinese_font = self._get_chinese_font()

        # Use font path saved in __init__.py directly
        if zh_font_paths and chinese_font in zh_font_paths:
            return FontProperties(fname=zh_font_paths[chinese_font])

        # Fallback to using font name
        return FontProperties(family=chinese_font)

    def _apply_chinese_font_to_figure(
        self,
        fig: Any,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[List[str]] = None,
        font_sizes: Optional[Dict[str, Union[int, float]]] = None,
    ) -> None:
        """
        Apply Chinese font to all text elements of a matplotlib Figure.
        Call before plt.show() or plt.savefig().
        """
        fonts = font_sizes or {}
        font_prop = self._get_font_properties()

        def apply_font_to_text(text_obj, new_text=None, fontsize=None):
            """Helper: apply Chinese font to a single text object."""
            if text_obj is None:
                return
            if new_text is not None:
                text_obj.set_text(new_text)
            text_obj.set_fontproperties(font_prop)
            if fontsize is not None:
                text_obj.set_fontsize(fontsize)

        for ax in fig.get_axes():
            # Title
            if ax.title:
                apply_font_to_text(ax.title, title, fonts.get("title"))

            # X-axis label
            if ax.xaxis.label:
                apply_font_to_text(ax.xaxis.label, xlabel, fonts.get("xlabel"))

            # Y-axis label
            if ax.yaxis.label:
                apply_font_to_text(ax.yaxis.label, ylabel, fonts.get("ylabel"))

            # Tick labels
            tick_size_x = fonts.get("tick") or fonts.get("xtick")
            for label in ax.get_xticklabels():
                apply_font_to_text(label, fontsize=tick_size_x)

            tick_size_y = fonts.get("tick") or fonts.get("ytick")
            for label in ax.get_yticklabels():
                apply_font_to_text(label, fontsize=tick_size_y)

            # Legend
            legend = ax.get_legend()
            if legend:
                if legend_title is not None:
                    legend.set_title(legend_title)
                if legend.get_title():
                    legend.get_title().set_fontproperties(font_prop)
                    if fonts.get("legend_title"):
                        legend.get_title().set_fontsize(fonts["legend_title"])

                texts = legend.get_texts()
                if legend_labels is not None:
                    if len(legend_labels) != len(texts):
                        warnings.warn(
                            "legend_labels length mismatches legend entries; will truncate to the minimal length",
                            UserWarning,
                        )
                    for text_obj, lbl in zip(texts, legend_labels):
                        text_obj.set_text(lbl)
                for text_obj in texts:
                    text_obj.set_fontproperties(font_prop)
                    if fonts.get("legend_label"):
                        text_obj.set_fontsize(fonts["legend_label"])

            # All other text annotations
            for text in ax.texts:
                text.set_fontproperties(font_prop)

        # Figure-level suptitle
        if hasattr(fig, "_suptitle") and fig._suptitle:
            fig._suptitle.set_fontproperties(font_prop)

    @contextmanager
    def _chinese_font_context(
        self,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[List[str]] = None,
        font_sizes: Optional[Dict[str, Union[int, float]]] = None,
    ):
        """
        Context manager: intercept plt.show() and plt.savefig(), apply Chinese font before execution.
        """
        import matplotlib.pyplot as plt

        original_show = plt.show
        original_savefig = plt.savefig
        original_figure_savefig = plt.Figure.savefig

        def intercepted_show(*args, **kwargs):
            fig = plt.gcf()
            self._apply_chinese_font_to_figure(
                fig, title, xlabel, ylabel, legend_title, legend_labels, font_sizes
            )
            return original_show(*args, **kwargs)

        def intercepted_savefig(fname, *args, **kwargs):
            fig = plt.gcf()
            self._apply_chinese_font_to_figure(
                fig, title, xlabel, ylabel, legend_title, legend_labels, font_sizes
            )
            return original_savefig(fname, *args, **kwargs)

        def intercepted_figure_savefig(self_fig, fname, *args, **kwargs):
            self._apply_chinese_font_to_figure(
                self_fig, title, xlabel, ylabel, legend_title, legend_labels, font_sizes
            )
            return original_figure_savefig(self_fig, fname, *args, **kwargs)

        try:
            plt.show = intercepted_show
            plt.savefig = intercepted_savefig
            plt.Figure.savefig = intercepted_figure_savefig
            yield
        finally:
            plt.show = original_show
            plt.savefig = original_savefig
            plt.Figure.savefig = original_figure_savefig

    def _apply_plotly_chinese_font(
        self,
        fig: Any,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> Any:
        """
        Detect and handle Chinese font settings for Plotly charts.

        If returned fig is a Plotly Figure object, set its font to Chinese font from zh_font_available.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return fig

        if not isinstance(fig, go.Figure):
            return fig

        # Get Chinese font
        if zh_font_available and isinstance(zh_font_available, list) and len(zh_font_available) > 0:
            chinese_font = zh_font_available[0]
        else:
            chinese_font = "Arial"

        # Update global font settings
        fig.update_layout(
            font=dict(family=chinese_font),
        )

        # Update title if specified
        if title is not None:
            fig.update_layout(
                title=dict(text=title, font=dict(family=chinese_font)),
            )

        # Update axis labels if specified
        if xlabel is not None:
            fig.update_xaxes(title=dict(text=xlabel, font=dict(family=chinese_font)))
        if ylabel is not None:
            fig.update_yaxes(title=dict(text=ylabel, font=dict(family=chinese_font)))

        return fig
