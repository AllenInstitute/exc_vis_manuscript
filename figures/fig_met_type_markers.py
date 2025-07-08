import argschema as ags
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score


MET_TYPE_ORDER = [
	"L2/3 IT",
	"L4 IT",
	"L4/L5 IT",
	"L5 IT-1",
	"L5 IT-2",
	"L5 IT-3 Pld5",
	"L6 IT-1",
	"L6 IT-2",
	"L6 IT-3",
	"L5/L6 IT Car3",
	"L5 ET-1 Chrna6",
	"L5 ET-2",
	"L5 ET-3",
	"L5 NP",
	"L6 CT-1",
	"L6 CT-2",
	"L6b",
]


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
    'L6b': '#2B7880',
}

MARKERS_TO_HIGHLIGHT = [
    "Pld5",
    "Chrna6",
]


# From Tasic et al. 2018 - Extended Data Fig. 5b
MARKER_GENES = [
    "Slc17a7",
    "Rtn4rl2",
    "Slc30a3",
    "Cux2",
    "Stard8",
    "Otof",
    "Rrad",
    "Adamts2",
    "Agmat",
    "Emx2",
    "Sla",
    "Ptrf",
    "Macc1",
    "Lrg1",
    "Rorb",
    "Rspo1",
    "Bglap3",
    "Scnn1a",
    "Endou",
    "Whrn",
    "Fezf2",
    "Hsd11b1",
    "Tox2",
    "Batf3",
    "Col6a1",
    "Col27a1",
    "Colq",
    "Ucma",
    "Tcap",
    "Olfr78",
    "Postn",
    "Npw",
    "Pld5",
    "Cbln4",
    "Cyp26b1",
    "Efemp1",
    "Wfdc18",
    "Htr2c",
    "Lypd1",
    "B3gnt8",
    "Egln3",
    "Scn7a",
    "Gpc3",
    "Cd72",
    "Wfdc17",
    "Ccdc42",
    "Rgs13",
    "Fgf17",
    "Erbb4",
    "Tdo2",
    "Gpr88",
    "Tnc",
    "Tmem163",
    "Dmrtb1",
    "Glipr1",
    "Arhgap25",
    "Aldh1a7",
    "Rxfp2",
    "Cpa6",
    "Gkn1",
    "Pcdh19",
    "Tgfb1",
    "Prss35",
    "Ctsc",
    "Oprk1",
    "Tunar",
    "Osr1",
    "Ppapdc1a",
    "Penk",
    "Fst",
    "Col23a1",
    "Col18a1",
    "Car3",
    "Car1",
    "Fam84b",
    "Chrna6",
    "Shisa3",
    "Erg",
    "Tph2",
    "Lgr5",
    "Tac1",
    "Pvalb",
    "C1ql2",
    "Ptgfr",
    "Stac",
    "Krt80",
    "Pappa2",
    "Slco2a1",
    "Lrrc9",
    "Dppa1",
    "Depdc7",
    "Npsr1",
    "Hpgd",
    "Sla2",
    "Trhr",
    "Slc17a8",
    "Rapgef3",
    "Met",
    "Trh",
    "Foxp2",
    "Syt6",
    "Nxph2",
    "Wls",
    "C1qtnf1",
    "Irf8",
    "Gpr139",
    "Defb1",
    "Ctxn3",
    "Ctgf",
    "Nxph4",
    "Fam150a",
    "Mup5",
    "Ngf",
    "Rxfp1",
    "Fbxl7",
    "F2r",
    "Olfr111",
    "Olfr110",
    "Sdc1",
    "Serpinb11",
    "P2ry12",
    "Kynu",
    "Crh",
    "Hsd17b2",
    "Mup3",
    "Lhx5",
    "Trp73",
    "Reln",
    "Cdh13",
    "Cpne7",
    "Brinp3",
    "Rprm",
    "Ntn5",
    "Efna1",
    "Nefl",
    "Efr3a",
]


class FigMetTypeMarkersParameters(ags.ArgSchema):
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    inf_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met type text labels",
    )
    output_file = ags.fields.OutputFile()


def main(args):
    tx_data_df = pd.read_feather(args['ps_tx_data_file'],
        columns=["sample_id"] + MARKER_GENES).set_index("sample_id")
    tx_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    tx_anno_df["spec_id_label"] = pd.to_numeric(tx_anno_df["spec_id_label"])
    tx_anno_df.set_index("spec_id_label", inplace=True)

    inf_met_type_df = pd.read_csv(args['inf_met_type_file'], index_col=0)

    avg_expr = {}
    pct_expr = {}
    for met_type in MET_TYPE_ORDER:
        spec_ids = inf_met_type_df.index[inf_met_type_df['inferred_met_type'] == met_type]
        print(met_type, len(spec_ids))

        sample_ids = tx_anno_df.loc[spec_ids, "sample_id"].values
        met_data = tx_data_df.loc[sample_ids, :]

        avg_expr[met_type] = met_data.mean(axis=0)
        pct_expr[met_type] = (met_data > 0).mean(axis=0)
    avg_expr_df = pd.DataFrame(avg_expr)
    pct_expr_df = pd.DataFrame(pct_expr)

    fig = plt.figure(figsize=(5, 10.5))
    gs = gridspec.GridSpec(5, 2, width_ratios=(1, 12), wspace=.6, hspace=1)

    scatter_factor = 15

    ax = plt.subplot(gs[:, 1])
    cbar_ax = plt.subplot(gs[0, 0])
    for i, g in enumerate(avg_expr_df.index):
        max_val = avg_expr_df.loc[g, :].max()
        max_pct_val = pct_expr_df.loc[g, :].max()
        sct = ax.scatter(
            x=np.arange(avg_expr_df.shape[1]),
            y=[i] * avg_expr_df.shape[1],
            c=avg_expr_df.loc[g, :] / max_val,
            s=pct_expr_df.loc[g, :] * scatter_factor,
            vmin=0,
            vmax=1,
            edgecolors='black',
            linewidths=0.25,
            cmap='RdYlBu_r',
        )
        if i == 0:
            cb = plt.colorbar(sct, cax=cbar_ax)

        ax.text(
            avg_expr_df.shape[1] + 2.5,
            i,
            f"{max_val:.1f}",
            va='center',
            ha='right',
            fontsize=5,
       )
    ax.text(
        avg_expr_df.shape[1] + 0.2,
        -1.2,
        "max. value (CPM)",
        va='center',
        ha='left',
        fontsize=5,
   )

    ax.set_yticks(range(avg_expr_df.shape[0]))
    ax.set_yticklabels(avg_expr_df.index, fontsize=9)
    for lab in ax.get_yticklabels():
       if lab.get_text() in MARKERS_TO_HIGHLIGHT:
          lab.set_fontweight('bold')

    ax.set_xticks(range(avg_expr_df.shape[1]))
    ax.set_xticklabels(
        [t for t in avg_expr_df.columns], rotation=90)
    ax.set_ylim(-0.5, avg_expr_df.shape[0] - 0.5)
    ax.set_xlim(-1, avg_expr_df.shape[1])
    ax.invert_yaxis()
    ax.tick_params("both", labelsize=5)

    cb.set_ticks([0, 1])
    cb.set_ticklabels(["0", "max"])
    cb.outline.set_visible(False)
    cbar_ax.tick_params(labelsize=7)
    cbar_ax.set_title("gene expression", fontsize=7)

    dot_ax = plt.subplot(gs[1, 0])

    dot_ax.scatter(
        x=[0] * 4,
        y=[0, 1, 2, 3],
        s=np.array([1.0, 0.5, 0.25, 0.1]) * scatter_factor,
        color='black',
        edgecolors='black',
        linewidths=0.25,
    )
    dot_ax.set_ylim(-5.5, 3.5)
    dot_ax.set(xticks=[], yticks=[])
    for y, t in zip((0, 1, 2, 3), ("1.0", "0.5", "0.25", "0.1")):
        dot_ax.text(0.1, y, t, fontsize=7, va='center', ha='left')
    dot_ax.set_title("proportion of cells", fontsize=7)
    sns.despine(ax=dot_ax, left=True, bottom=True)

    sns.despine(ax=ax, left=True, bottom=True)
    plt.savefig(args["output_file"], dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigMetTypeMarkersParameters)
    main(module.args)
