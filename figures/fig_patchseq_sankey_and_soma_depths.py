import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import json

from mouse_met_figs.simple_sankey import sankey
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
import argschema as ags

matplotlib.rc('font', family='Helvetica')
sns.set(style="ticks", context="paper", font="Helvetica", rc={"grid.linewidth": 0.5}, font_scale=1.3)
sns.set_palette("bright")


class PlotSankeyandDepthViolinParameters(ags.ArgSchema):
    ps_features_file = ags.fields.InputFile(
        description="csv file with Patch-seq morphology features. Must include 'soma_aligned_dist_from_pia'",
        allow_none=False
        )
    ps_flatmap_coords_file = ags.fields.InputFile(
        description="csv file with flat map coordinates from CCF location of Patch-seq cells.",
        allow_none=False
    )
    ps_met_types_file = ags.fields.InputFile(
        description="csv file with MET-type assignments for Patch-seq cells. Must include ''",
        allow_none=False
    )
    ps_specimens_file = ags.fields.InputFile(
        description="txt file with Patch-seq specimen IDs",
        allow_none=True,
        default=None
        )
    ps_anno_file = ags.fields.InputFile(
        description="feather file with Patch-seq transcriptomic annotations (for ttype colors)",
        allow_none=False
        )
    wnm_features_file = ags.fields.InputFile(
        description="csv file with Whole Neuron Morphology cell morphology features. Must include 'soma_aligned_dist_from_pia'",
        allow_none=False
    )
    wnm_met_types_file = ags.fields.InputFile(
        description="csv file with MET-type assignments for Whole Neuron Morphology cells. Must include 'inferred_met_type'",
        allow_none=False
    )
    layer_depths_file = ags.fields.InputFile(
        description="json file with layer depths",
        allow_none=False
    )
    output_file = ags.fields.OutputFile(
        description="output figure file name"
    )
    log_level = ags.fields.LogLevel(
        default='ERROR',
        description="set the logging level of the module")
    

def split_ttype_name(label, max_len_label, remove_visp=True):
    if remove_visp == True:
        label = label.replace(" VISp "," ")
    if len(label) > max_len_label:
        num_words = len(label.split(" "))
        if num_words > 3:
            mod_label =  F"{' '.join(label.split(' ')[0:3])}\n{' '.join(label.split(' ')[3::])}"
        else:
            mod_label =  F"{' '.join(label.split(' ')[0:2])}\n{' '.join(label.split(' ')[2::])}"

    else:
        mod_label = label
    return mod_label

   
def main(ps_features_file, ps_flatmap_coords_file, ps_met_types_file, ps_specimens_file, ps_anno_file,
         wnm_features_file, wnm_met_types_file, layer_depths_file,
         output_file, log_level, **kwargs):

    # Load the data
    # Patch-seq specimen list
    with open(ps_specimens_file, "r") as fn:
        ps_specimens = [int(x) for x in fn.readlines()]

    # Patch-seq flat map coordinates
    ps_flatmap_coords_df = pd.read_csv(ps_flatmap_coords_file, index_col=0)
    
    # Patch-seq and Whole Neuron Morphology morphology feature files
    ps_features_df = pd.read_csv(ps_features_file, index_col=0)
    wnm_features_df = pd.read_csv(wnm_features_file, index_col="swc_path")

    # Patch-seq MET assignments
    ps_met_types_df = pd.read_csv(ps_met_types_file, index_col=0)
    # Rename PT as ET
    for col in ["t_type","met_type","inferred_met_type"]:
        ps_met_types_df.loc[:,col] = ps_met_types_df[col].apply(lambda x: x.replace("PT","ET") if type(x) != float else x)
    ps_met_types_df["final_met_type"] = ps_met_types_df["inferred_met_type"].fillna("undetermined")
    
    # Whole Neuron Morphology MET assignments
    wnm_met_types_df = pd.read_csv(wnm_met_types_file, index_col=0)
    wnm_met_types_df.index = list(map(lambda x: x.replace(".swc",""), wnm_met_types_df.index))

    # Layer depths file
    with open(layer_depths_file, "r") as f:
        layer_info = json.load(f)

    # Patch-seq transcriptomic annotations
    ps_anno_df = pd.read_feather(ps_anno_file)
    ps_anno_df.set_index(keys="spec_id_label", inplace=True)
    ps_anno_df["cluster_label"] = ps_anno_df["cluster_label"].apply(lambda x: x.replace("PT","ET") if type(x) != float else x)

    # Combine depths from the two datasets
    ps_tmp = ps_flatmap_coords_df["depth"].to_frame().join(ps_features_df["soma_aligned_dist_from_pia"], how="left")
    ps_tmp.rename(columns={"depth":"depth_from_flatmap"}, inplace=True)
    ps_ids = list(set(ps_specimens) & set(ps_tmp.index))
    ps_tmp = ps_tmp.loc[ps_ids, :]
    ps_tmp["met_type"] = ps_met_types_df.loc[ps_ids,"inferred_met_type"].values
    ps_tmp["dataset"]="Patch-seq"
    
    wnm_tmp = wnm_features_df["soma_aligned_dist_from_pia"].to_frame()
    wnm_tmp = wnm_tmp.join(wnm_met_types_df["predicted_met_type"], how="left")
    wnm_tmp.rename(columns={"predicted_met_type":"met_type"}, inplace=True)
    wnm_tmp["dataset"] = "Whole Neuron Morphology"

    depth_df = pd.concat([ps_tmp, wnm_tmp], axis=0)

    # Where possible, take depth from reconstruction for Patch-seq cell
    depth_df["combined_depth"] = depth_df["soma_aligned_dist_from_pia"].copy()
    recon_ids = depth_df[depth_df["soma_aligned_dist_from_pia"].notnull()].copy().index.tolist()
    no_recon_ids = depth_df[depth_df["soma_aligned_dist_from_pia"].isnull()].copy().index.tolist()
    depth_df.loc[no_recon_ids,"combined_depth"] = depth_df.loc[no_recon_ids,"depth_from_flatmap"].values


    # Setup for figure
    MET_TYPE_COLORS = {
        'L2/3 IT': '#7AE6AB',
        'L4 IT': '#00979D',
        'L4/L5 IT': '#00DDC5',
        'L5 IT-1': '#00A809',
        'L5 IT-2': '#00FF00',
        'L5 IT-3 Pld5': '#26BF64',
        'L6 IT-1': '#C2E32C',
        'L6 IT-2': '#96E32C',
        'L6 IT-3': '#A19922',
        'L5/L6 IT Car3': '#5100FF',
        'L5 ET-1 Chrna6': '#0000FF',
        'L5 ET-2': '#22737F',
        'L5 ET-3': '#29E043',
        'L5 NP': '#73CA95',
        'L6 CT-1': '#74CAFF',
        'L6 CT-2': '#578EBF',
        'L6b': '#2B7880'
        }
    # Ordering for figure
    dataset_order = [
        "Patch-seq", "Whole Neuron Morphology"
        ]
    met_type_order = [
        'L2/3 IT', 'L4 IT', 'L4/L5 IT', 
        'L5 IT-1', 'L5 IT-2', 'L5 IT-3 Pld5',
        'L6 IT-1', 'L6 IT-2','L6 IT-3', 'L5/L6 IT Car3', 
        'L5 ET-1 Chrna6', 'L5 ET-2', 'L5 ET-3',
        'L5 NP', 
        'L6 CT-1', 'L6 CT-2', 
        'L6b'
        ]
    ttype_order = [
        'L2/3 IT VISp Adamts2',
        'L2/3 IT VISp Rrad',
        'L2/3 IT VISp Agmat',
        'L4 IT VISp Rspo1',
        'L5 IT VISp Whrn Tox2',
        'L5 IT VISp Hsd11b1 Endou',
        'L5 IT VISp Batf3',
        'L5 IT VISp Col6a1 Fezf2',
        'L5 IT VISp Col27a1',
        'L6 IT VISp Col23a1 Adamts2',
        'L6 IT VISp Col18a1',
        'L6 IT VISp Penk Fst',
        'L6 IT VISp Penk Col27a1',
        'L6 IT VISp Car3',
        'L5 ET VISp Chrna6',
        'L5 ET VISp Lgr5',
        'L5 ET VISp Krt80',
        'L5 ET VISp C1ql2 Cdh13',
        'L5 ET VISp C1ql2 Ptgfr',
        'L5 NP VISp Trhr Met',
        'L5 NP VISp Trhr Cpne7',
        'L6 CT VISp Ctxn3 Brinp3',
        'L6 CT VISp Nxph2 Wls',
        'L6 CT VISp Ctxn3 Sla',
        'L6 CT VISp Gpr139',
        'L6 CT VISp Krt80 Sla',
        'L6b VISp Crh',
        'L6b P2ry12',
        'L6b VISp Col8a1 Rxfp1',
        'L6b VISp Mup5',
        'L6b Col8a1 Rprm'
        ]
    met_types_ordering_dict = {val:i for i,val in enumerate(met_type_order)}

    combined_ordering = {}
    for i,met in enumerate(met_type_order):
        for j,ds in enumerate(dataset_order):
            combined_ordering[F"{met} {ds}"] = (2*i)+j


    # Palette
    ttype_palette = ps_anno_df.set_index("cluster_label")["cluster_color"].to_dict()
    combo_palette = {}
    combo_palette.update(ttype_palette)
    combo_palette.update(MET_TYPE_COLORS)

    # Plot depth histograms with river plot
    # Sankey constants 
    orders = sorted(met_types_ordering_dict.values())
    rev_met_types_ordering_dict = {v:k for k,v in met_types_ordering_dict.items()}
    left_order = [rev_met_types_ordering_dict[x] for x in orders]
    fs=8
    min_cells_for_label = 1
    center_right = False

    # Violin constants
    violin_width = .8
    jitter = 0.15
    depth_var = "combined_depth"


    fig = plt.figure(figsize=(14,3.))
    g_main = gridspec.GridSpec(ncols=1,nrows=2, height_ratios=(.5,1), hspace=.45)
    g_violin = gridspec.GridSpecFromSubplotSpec(ncols=3, nrows=1, width_ratios=(.06,1.,.1), wspace=0, subplot_spec=g_main[1])
    sankey_ax = plt.subplot(g_main[0])
    violin_ax = plt.subplot(g_violin[1])

    # Plot Sankey
    right_col = "t_type"
    left_col = "met_type"
    sankey_df = ps_met_types_df.dropna(subset=["met_type"])
    max_len_label = 10

    (l_labels, r_labels) = sankey(
        left=sankey_df[left_col].tolist(),
        right=sankey_df[right_col].tolist(),
        colorDict=combo_palette,
        leftLabels=left_order,
        rightLabels=ttype_order,
        aspect=1,
        rightColor=True,
        fontsize=fs,
        returnLabels=True,
        rearrange=False,
        orientation='horizontal',
        ax=sankey_ax
    )
    for llab in l_labels:
        label_text = llab['text'].replace(" ","\n")
        sankey_ax.text(
            x=llab["x"], 
            y=llab["y"]-20, 
            s=label_text, 
            fontdict={
                    'ha': 'center', 
                    'va': 'top', 
                    'fontsize': fs+1, 
                    'rotation':'horizontal', 
                    'color': MET_TYPE_COLORS[llab["text"]]
                }
        )

    mapped_type_counts = sankey_df[right_col].value_counts().to_dict()
    for rlab in r_labels:
        ttype_label = rlab["text"]
        split_modified_ttype_label = split_ttype_name(label=ttype_label, max_len_label=max_len_label, remove_visp=True)
        
        rcount = mapped_type_counts[rlab['text']]
        if rcount >= min_cells_for_label:
            label_text = F"{split_modified_ttype_label} ({mapped_type_counts[rlab['text']]})"
            if center_right == True:
                ha="center"
                shift = 850
            else:
                ha="left"
                shift = 20

            if "\n" in label_text:
                double_line_shift = 4
            else:
                double_line_shift = 1

            sankey_ax.text(
                x=rlab["x"]-double_line_shift, 
                y=rlab["y"]+20, 
                s=label_text, 
                fontdict={
                    'ha': 'left', 
                    'va': 'bottom', 
                    'fontsize': fs, 
                    'rotation':'vertical', 
                    'color': 'black',
                    'linespacing':.9
                }
            )
        

    # Plot violins
    g = sns.violinplot(
        data=depth_df,
        x="met_type",
        y=depth_var,
        order=met_type_order,
        cut=0,
        hue="dataset",
        hue_order=dataset_order,
        split=True,
        width=violin_width,
        density_norm="width",
        inner=None,
        linewidth=0.5,
        legend=False,
        ax=violin_ax
    )

    sns.stripplot(
        data=depth_df,
        x="met_type",
        y=depth_var,
        order=met_type_order,
        hue="dataset",
        hue_order=dataset_order,
        dodge=True,
        size=2,
        palette={"Patch-seq": "#000000", "Whole Neuron Morphology": "#666666"},
        jitter=jitter,
        legend=False,
        ax=violin_ax
    )

    # Add layer lines
    violin_ax.axhline(0, linestyle="solid", color="lightgray", zorder=-1, lw=1)
    for k, v in layer_info.items():
        violin_ax.axhline(v, linestyle="solid", color="lightgray", zorder=-1, lw=1)
    # Add layer labels
    layer_label_info = {
        "1": np.mean([0, layer_info["2/3"]]),
        "2/3": np.mean([layer_info["2/3"], layer_info["4"]]),
        "4": np.mean([layer_info["4"], layer_info["5"]]),
        "5": np.mean([layer_info["5"], layer_info["6a"]]),
        "6a": np.mean([layer_info["6a"], layer_info["6b"]]),
        "6b": np.mean([layer_info["6b"], layer_info["wm"]]),
    }
    trans = violin_ax.get_yaxis_transform()
    for l, v in layer_label_info.items():
        violin_ax.text(
            1.018,
            v, "L"+l,
            fontsize=fs, ha="left", va="center", color="black",
            transform=trans)


    # The color changing method below doesn't quite work as there are places where no violin is drawn for that met-type dataset combo (ie zero cells)
    # Getting the drawn labels by counting how many cells are in each bucket.
    counts_df = depth_df.groupby(by=[pd.Categorical(depth_df.met_type),"dataset"]).agg({depth_var:"count"}).fillna(0).reset_index()
    counts_df.rename(columns={"level_0":"met_type", depth_var:"count"}, inplace=True)
    counts_df["met_ds"] = counts_df.apply(lambda x: F"{x['met_type']} {x['dataset']}", axis=1)
    counts_df["sort_order"] = counts_df["met_ds"].map(combined_ordering)
    counts_df.sort_values(by="sort_order", inplace=True)

    # Take highest val for each met-type for sample size label
    sample_size_label_yoffset = -70
    sample_size_label_ypos = depth_df.groupby(by="met_type").agg({"combined_depth":"min"}) + sample_size_label_yoffset

    violin_met_labels = counts_df[counts_df["count"] > 1].copy()["met_type"].values.tolist()
    violin_ds_labels = counts_df[counts_df["count"] > 1].copy()["dataset"].values.tolist()

    # Recolor violins - https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot
    for ind, violin in enumerate(violin_ax.findobj(PolyCollection)):
        current_met = violin_met_labels[ind]
        current_ds = violin_ds_labels[ind]
        label_ypos = sample_size_label_ypos.loc[current_met].values[0]
        
        if current_ds == "Patch-seq":
            violin_xmin = np.min(list(zip(*violin.get_paths()[0].vertices))[0]).round(1)
            violin_xmax = np.max(list(zip(*violin.get_paths()[0].vertices))[0]).round(1)
            label_xpos1 = violin_xmin
            label_xpos2 = violin_xmax+(violin_xmax-violin_xmin)
            
            # Pad numbers so that they are centered on the center of the violin
            ps_n = counts_df[(counts_df["met_type"]==current_met) & (counts_df["dataset"]=="Patch-seq")]["count"].values[0]
            ps_n_str = str(ps_n)
            
            wnm_n = counts_df[(counts_df["met_type"]==current_met) & (counts_df["dataset"]=="Whole Neuron Morphology")]["count"].values[0]
            wnm_n_str = str(wnm_n)
            
            violin_ax.text(
                label_xpos1+0.05, 
                y=label_ypos, 
                s=F"{ps_n}",
                fontdict={
                "fontsize":fs,
                "ha":"left"
                }
            )
            violin_ax.text(
                label_xpos2-0.05, 
                y=label_ypos, 
                s=F"{wnm_n}",
                fontdict={
                "fontsize":fs,
                "ha":"right"
                }
            )
        
        rgb = colors.to_rgb(MET_TYPE_COLORS[current_met])
        if current_ds == "Whole Neuron Morphology":
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)


    violin_ax.set_ylim([-10,1100]);
    violin_ax.invert_yaxis()
    violin_ax.set_xticks([]);
    violin_ax.set_yticks([]);
    violin_ax.set_xlabel("");
    violin_ax.set_ylabel("");
    sns.despine(ax=violin_ax, top=True, right=True, bottom=True, left=True, trim="True");


    # Create scale bar
    scalebar = ScaleBar(
        dx=1.,
        fixed_value=200,
        units="um",
        scale_loc="right",
        width_fraction=.004,
        border_pad=0,
        pad=0,
        label_formatter = lambda x, y:'', # removes the text label
        frameon=False,
        rotation="vertical",
        bbox_transform=violin_ax.transAxes,
        bbox_to_anchor=(1.025, 0.9),
        font_properties={
            "size":fs,
        },
        color="black"
    )

    violin_ax.add_artist(scalebar)


    ps_recon_ids = depth_df[(depth_df["dataset"]=="Patch-seq") & (depth_df["soma_aligned_dist_from_pia"].notnull())].copy().index.tolist()
    ps_no_recon_ids = depth_df[(depth_df["dataset"]=="Patch-seq") & (depth_df["soma_aligned_dist_from_pia"].isnull())].copy().index.tolist()
    print(F"Cells in soma depth histogram:\n{depth_df[depth_df[depth_var].notnull()].copy()['dataset'].value_counts()}")
    print(F"{len(ps_recon_ids)} Patch-seq cells with soma depth acquired from reconstruction.")
    print(F"{len(ps_no_recon_ids)} Patch-seq cells with soma depth acquired from flatmap.")

    plt.savefig(output_file, dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=PlotSankeyandDepthViolinParameters)
    main(**module.args)
    

