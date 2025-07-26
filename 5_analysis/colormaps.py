from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex

# Custom Colors
_COLORS = ["#FFFFFF", "#9988FF", "#292828"]


def get_continuous_cmap_white(n_colors: int = 256):
    """
    Returns a continuous LinearSegmentedColormap with thesis theme.
    """
    return LinearSegmentedColormap.from_list("thesis_theme", _COLORS, N=n_colors)


def get_segmented_cmap(n_segments: int = 7, hex=True):
    """
    Returns a ListedColormap with n_segments colors taken from the thesis theme.
    """
    continuous = get_continuous_cmap_white()
    color_list = [continuous(i / (n_segments - 1)) for i in range(n_segments)]
    if hex:
        hex_colors = [to_hex(c) for c in color_list]
        return hex_colors
    else:
        return ListedColormap(color_list, name=f"thesis_segmented_{n_segments}")


def get_continuous_cmap_lilac(n_colors: int = 256):
    """
    Returns a continuous LinearSegmentedColormap with thesis theme without white.
    """
    seg_cmap = get_segmented_cmap(5, hex=False)
    color_list = [seg_cmap(i) for i in range(1, seg_cmap.N)]
    return LinearSegmentedColormap.from_list(
        "thesis_theme_lilac", color_list, N=n_colors
    )
