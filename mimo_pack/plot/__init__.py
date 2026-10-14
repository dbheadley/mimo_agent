import mimo_pack.plot.colors  # registers 'mimo_sequential' and 'mimo_diverging' colormaps

from mimo_pack.plot.colors import MIMO_PALETTE, MIMO_SEQ_CMAP, MIMO_DIV_CMAP
from mimo_pack.plot.connectivity_web import connectivity_page, connectivity_index_page
from mimo_pack.plot.dataframe import PlotGrid
from mimo_pack.plot.probe import (
    probe_layout,
    probe_layout_subplots,
    channel_values,
    depth_lines,
    unit_lfp_mapping,
)
