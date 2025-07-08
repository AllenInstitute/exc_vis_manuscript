import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys


def basic_morph_plot(morph, ax, morph_colors={3: "firebrick", 4: "salmon", 2: "steelblue"},
                     side=False, xoffset=0, alpha=1.0):
    for compartment, color in morph_colors.items():
        lines_x = []
        lines_y = []
        for c in morph.compartment_list_by_type(compartment):
            if c["parent"] == -1:
                continue
            p = morph.compartment_index[c["parent"]]
            if side:
                lines_x += [p["z"] + xoffset, c["z"] + xoffset, None]
            else:
                lines_x += [p["x"] + xoffset, c["x"] + xoffset, None]
            lines_y += [p["y"], c["y"], None]
        ax.plot(lines_x, lines_y, c=color, linewidth=0.25, zorder=compartment, alpha=alpha)
    return ax


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
