
from skeleton_keys.io import load_default_layer_template
import shutil
import ccf_streamlines.morphology as ccfmorph
import ccf_streamlines.projection as ccfproj
from ccf_streamlines.angle import coordinates_to_voxels
from ccf_streamlines.coordinates import coordinates_to_voxels
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from pathlib import Path
from ccf_streamlines.angle import find_closest_streamline
import os
import json
import seaborn as sns
import re
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import json 
# from allensdk.core.reference_space_cache import ReferenceSpaceCache
from itertools import combinations
import random
from scipy import stats
from statsmodels.stats.multitest import multipletests
# from statannot import add_stat_annotation
import scikit_posthocs as sci_po
from collections import defaultdict
from statannotations.Annotator import Annotator

from morph_utils.ccf import load_structure_graph
# from statannot import add_stat_annotation
import scikit_posthocs as sp
from collections import defaultdict
from statannotations.Annotator import Annotator

from morph_utils.ccf import load_structure_graph, open_ccf_annotation
from skeleton_keys.io import load_default_layer_template
from tqdm import tqdm
from copy import copy
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from scipy import stats


def main():
    
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)

    with open(args['color_file'],'r') as f:
        color_dict=json.load(f)


    labels_df = pd.read_csv(args['fmost_metadata_file'],index_col=0)
    fmost_feats = pd.read_csv(args['fmost_raw_local_dend_feature_file'],index_col=0)
    soma_depth_by_id = fmost_feats['soma_aligned_dist_from_pia'].to_dict()



    projection_matrix_roll_up_file = args['fmost_projection_matrix_roll_up']

    proj_df_rollup = pd.read_csv(projection_matrix_roll_up_file,index_col=0)
    # proj_df_rollout = pd.read_csv(projection_matrix_roll_out_file,index_col=0)
    # proj_df_rollout.index = proj_df_rollout.index.map(lambda x:os.path.basename(x))

    roll_up_proj_cols = [c for c in proj_df_rollup.columns if any([i in c for i in ["ipsi",'contra']]) and "fiber" not in c]
    # roll_out_proj_cols = [c for c in proj_df_rollout.columns if any([i in c for i in ["ipsi",'contra']]) and "fiber" not in c]



    # cortical_acs = [d['acronym'] for d in cortical_descendants]
    sg_df = load_structure_graph()
    ctx_id = sg_df.loc['CTX'].id
    cortical_acs = sg_df[sg_df.structure_id_path.apply(lambda x: ctx_id in x)].index.tolist()

    ipsis_ctx = ["ipsi_{}".format(c) for c in cortical_acs]
    # ipsis_ctx = [c for c in ipsis_ctx if c in roll_up_proj_cols]

    contras_ctx = ["contra_{}".format(c) for c in cortical_acs]
    # contras_ctx = [c for c in contras_ctx if c in roll_up_proj_cols]

    cortical_proj_cols = ipsis_ctx + contras_ctx



    proj_df_rollup = proj_df_rollup.merge(labels_df,left_index=True,right_index=True)
    # proj_df_rollout = proj_df_rollout.merge(labels_df,left_index=True,right_index=True)

    proj_df_rollup["total_ipsi_ctx_length"] = proj_df_rollup[[c for c in ipsis_ctx if c in proj_df_rollup.columns]].sum(axis=1)
    proj_df_rollup["total_contra_ctx_length"] = proj_df_rollup[[c for c in contras_ctx if c in proj_df_rollup.columns]].sum(axis=1)
    proj_df_rollup["projects_to_contra_ctx"] = proj_df_rollup[[c for c in contras_ctx if c in proj_df_rollup.columns]].sum(axis=1).astype(bool)

    proj_df_rollup['total_axon_projection_length'] = proj_df_rollup[roll_up_proj_cols].sum(axis=1)
    proj_df_rollup['fraction_of_total_axon_in_contra_ctx'] = proj_df_rollup["total_contra_ctx_length"]/proj_df_rollup['total_axon_projection_length']

    total_contra_length_by_id = proj_df_rollup['total_contra_ctx_length'].to_dict()

    met_type_value_counts = proj_df_rollup['predicted_met_type'].value_counts().to_dict()

    proportion_df = pd.DataFrame(proj_df_rollup.groupby("predicted_met_type")['projects_to_contra_ctx'].mean())

    proportion_df = proportion_df.reset_index()
    it_proportions = proportion_df[(proportion_df['predicted_met_type'].str.contains("IT")) | 
                                    proportion_df['predicted_met_type'].str.contains("L6b")
                                    ]
    it_proportions = it_proportions[it_proportions['projects_to_contra_ctx']>0]
    it_proportions = it_proportions[it_proportions['predicted_met_type'].map(met_type_value_counts)>2]


    it_met_types = sorted(it_proportions['predicted_met_type'].values)

    possible_combinations = list(combinations(it_proportions['predicted_met_type'],2))
    col = "Contra-CTX"
    monte_carlo_resdict = {"projection_target":col}

    verbose = False
    monte_carlo_records = []
    montecarlo_corrected_records = []
    pvals = []
    group_names = []
    for combo in possible_combinations:
        
        m1 = combo[0]
        m2 = combo[1]
        
        n1 = met_type_value_counts[m1]
        n2 = met_type_value_counts[m2]
        
        p1 = it_proportions[it_proportions['predicted_met_type']==m1]['projects_to_contra_ctx'].iloc[0]
        p2 = it_proportions[it_proportions['predicted_met_type']==m2]['projects_to_contra_ctx'].iloc[0]
        
        n1_positive = n1*p1
        n2_positive = n2*p2
        
        observed_difference = p2-p1
        
        # create a binary representation of the data pool
        n1_projection_array = [1]*int(n1_positive) + [0]*int(n1-n1_positive)
        n2_projection_array = [1]*int(n2_positive) + [0]*int(n2-n2_positive)
        data_pool = n1_projection_array + n2_projection_array

        # randomly create group 1 and group 2 on newly shuffled data pool
        num_its=10000
        random_differences = []
        for _ in range(num_its):
            random.shuffle(data_pool)
            data1_random = data_pool[0:n1]
            data2_random = data_pool[n1:]

            p1_random = sum(data1_random)/n1
            p2_random = sum(data2_random)/n2

            # record random difference from group 1 and group 2
            random_difference = p2_random-p1_random
            random_differences.append(random_difference)


        # see where the observed measurement lies in the simulated data distribution
        sorted_difference_values = sorted(random_differences+[observed_difference])
        percentile = stats.percentileofscore(sorted_difference_values, observed_difference)
        if percentile<50:
            monte_carlo_p_val = percentile*0.01
        else:
            monte_carlo_p_val = (100-percentile)*0.01

        monte_carlo_resdict["{} & {}".format(m1,m2)]=monte_carlo_p_val 


        group_names.append("{} & {}".format(m1,m2))
        pvals.append(monte_carlo_p_val)

        if verbose:
            plt.hist(random_differences)
            plt.axvline(observed_difference,c='r')
            plt.title(" ".join(combo)+ " p={}".format(monte_carlo_p_val))
            plt.show()
            plt.clf()
            
    monte_carlo_records.append(monte_carlo_resdict)

    print("running multiple comparisons for {} pvals".format(len(pvals)))
    # multiple comparison correction
    multitest_res = multipletests(pvals,alpha=0.05, method='fdr_bh')
    adjusted_pvals = multitest_res[1]
    montecarlo_corrected_results = {"projection_target":col}
    for group_name, adjust_pval in zip(group_names,adjusted_pvals):
        montecarlo_corrected_results[group_name]=adjust_pval
    montecarlo_corrected_records.append(montecarlo_corrected_results)

    montecarlo_corrected_resdf = pd.DataFrame.from_records(montecarlo_corrected_records)
    montecarlo_corrected_resdf = montecarlo_corrected_resdf.set_index("projection_target")

    montecarlo_corrected_resdf = montecarlo_corrected_resdf.T

    montecarlo_corrected_resdf_sig = montecarlo_corrected_resdf[montecarlo_corrected_resdf['Contra-CTX']<0.05]



    fig, axe = plt.gcf(), plt.gca()

    # Bar plot
    bar = sns.barplot(data=it_proportions, x='predicted_met_type', y='projects_to_contra_ctx', palette=color_dict)

    # Get group label positions on the x-axis
    xtick_labels = [tick.get_text() for tick in axe.get_xticklabels()]
    group_locs = {label: i for i, label in enumerate(xtick_labels)}

    # Get the max height of all bars
    max_height = max([patch.get_height() for patch in bar.patches])

    # Draw manual significance bars
    y_base = max_height + 0.02  # start just above the tallest bar
    line_spacing = 0.05       # space between stacked lines
    line_height = 0.01          # height of vertical ticks

    for idx, (group, row) in enumerate(montecarlo_corrected_resdf_sig.iterrows()):
        pval = row['Contra-CTX']
        g1, g2 = group.split(" & ")

        x1 = group_locs[g1]
        x2 = group_locs[g2]
        x_center = (x1 + x2) / 2

        y = y_base + idx * line_spacing

        # Draw significance bar
        axe.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], lw=0.5, c='black')

        if pval < 0.05:
            text = '*'
        else:
            text = 'n.s.'

        axe.text(x_center, y - 0.04 , text, ha='center', va='bottom', color='black',size=9)

    # Style the plot
    axe.set_xticklabels(axe.get_xticklabels(), rotation=90)
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)
    axe.set_xlabel("")
    axe.set_ylabel("Fraction of Cells")
    fig.set_size_inches(1.5, 1.5)
    fig.savefig("plot_SuppFig_ContraCellTypes_FractionITMETsContra.pdf",dpi=300,bbox_inches='tight')
    plt.close()
    


    proj_df_rollup['VISp_HVA'] = "HVA"
    proj_df_rollup.loc[proj_df_rollup.ccf_soma_location_nolayer=='VISp','VISp_HVA']= 'VISp'

    y_cols_to_plot = ['fraction_of_total_axon_in_contra_ctx', 'total_contra_ctx_length']
    y_config = {
        "Fract. Axon in Contra. Cortex": ["FractAxonInContraCtx.pdf", 'fraction_of_total_axon_in_contra_ctx' ],
        "Contra. Cortex Axon Length (μm)": ["ContraCortexAxonLen.pdf", 'total_contra_ctx_length']
    }


    for y_label, y_list in y_config.items():
        y = y_list[1]

        mets_of_interest = [ 'L4/L5 IT',   "L5/L6 IT Car3"]
        mets_of_interest_df = proj_df_rollup[proj_df_rollup['predicted_met_type'].isin(mets_of_interest)]
        mets_of_interest_contra_df = mets_of_interest_df[mets_of_interest_df["total_contra_ctx_length"]!=0]
        mets_of_interest_contra_df = mets_of_interest_contra_df.sort_values(by='predicted_met_type')
        f2,a2 = plt.gcf(),plt.gca()
        
        
        bplt = sns.boxplot(data = mets_of_interest_contra_df, 
                            x='predicted_met_type', y=y,
                            hue='VISp_HVA',
                        )  

        
        all_pvls = []
        all_groups = []

        a = []
        labels = []
        for mt, mt_df in mets_of_interest_contra_df.groupby('predicted_met_type'):
            
            for hue_val in mets_of_interest_contra_df['VISp_HVA'].unique():
                this_df = mt_df[mt_df['VISp_HVA']==hue_val]
                labels.append("{}__{}".format(mt,hue_val))
                feat = this_df[y].values.tolist()
                a.append(feat)

        ks, kp = stats.kruskal(*a)
        pval_thresh = 0.05
        sig_results = []
        if kp<0.05:
            dunn_df = sci_po.posthoc_dunn(a)
            dunn_df.index=labels
            dunn_df.columns=labels
            sig_diff_dunn = dunn_df[dunn_df<pval_thresh]
            sig_diff_dunn = sig_diff_dunn.mask(np.triu(np.ones(sig_diff_dunn.shape, dtype=np.bool_)))

            sig_results = list(sig_diff_dunn[sig_diff_dunn.notnull()].stack().index)
        
        if sig_results:
            for sig_pair in sig_results:
                pval = dunn_df.loc[sig_pair[0]][sig_pair[1]]
                all_pvls.append(pval)
                
                pair_frmt = (tuple(sig_pair[0].split("__")), tuple(sig_pair[1].split("__")))
                all_groups.append(pair_frmt)
        
        if all_groups:
                
            x = 'predicted_met_type'
            y = y
            annot = Annotator(ax=bplt, pairs=all_groups, data=mets_of_interest_contra_df, x=x, y=y, hue='VISp_HVA')
            annot.configure(line_offset = 30000)
            annot.set_pvalues(all_pvls)
            annot.annotate()

        a2.set_ylabel(y_label,size=8)
        a2.spines['top'].set_visible(False)
        a2.spines['right'].set_visible(False)
        a2.set_xticklabels(a2.get_xticklabels(), rotation=90,size=8)
        # a2.set_yticks([0,0.5,1])
        a2.set_yticklabels(a2.get_yticklabels(), size=8)
        a2.set_xlabel('')
        f2.set_size_inches(1.5,1.5)
        plt.legend(fontsize=6, handlelength=1, loc='center left', bbox_to_anchor=(0.75, 0.25))

        ofi = os.path.join("plot_SuppFig_ContraCellTypes_{}".format(y_list[0]))
        f2.savefig(ofi,dpi=300,bbox_inches='tight')
        plt.clf()
        plt.close()
    
    
    
    ####
    ####
    ####  Flatmap Slab/Histograms
    ####
    
    slab_csv_dir = args['slab_csv_dir']
    
    top_shape = (200, 2720)
    main_shape = (1360, 2720)
    
    def plot_hist_to_axis(mean_hist, sem_hist, ax, negative, color):

        if negative:
            mean_hist = -1*mean_hist
            
            
        y_vals = list(np.arange(len(mean_hist)))
        ax.plot(mean_hist, y_vals,c=color)
        ax.fill_betweenx(
            y_vals,
            mean_hist - sem_hist,
            mean_hist + sem_hist,
            color=color,
            alpha=0.3,
        )


    sg_df = load_structure_graph()
    ctx_id = sg_df.loc['CTX'].id
    cortical_acs = sg_df[sg_df.structure_id_path.apply(lambda x: ctx_id in x)].index.tolist()

    ipsis_ctx = ["ipsi_{}".format(c) for c in cortical_acs]
    contras_ctx = ["contra_{}".format(c) for c in cortical_acs]

    cortical_proj_cols = ipsis_ctx + contras_ctx

    proj_df = pd.read_csv(args['fmost_projection_matrix_roll_up'],index_col=0)

    proj_df['contra_cortex_amount'] = proj_df[[c for c in contras_ctx if c in proj_df.columns]].sum(axis=1)

    contra_projecting_ids = proj_df[proj_df['contra_cortex_amount'] != 0].index.tolist()
    contra_projecting_ids = set(contra_projecting_ids) & set(labels_df.index)



    contra_label_df = labels_df.loc[list(contra_projecting_ids)].copy()
    midline = int(top_shape[1]/2)
    normalize='log'
    import warnings
    warnings.filterwarnings('ignore')
    met_type_slab_records = {}
    for met_type, met_df in contra_label_df.groupby("predicted_met_type"):
        True
        full_met_df = labels_df[labels_df['predicted_met_type']==met_type]
        met_type_slab_records[met_type] = {}
        this_met_image = np.zeros((top_shape[0], top_shape[1], met_df.shape[0]))
        this_met_soma_locs = []
        cell_idx = -1
        for cell_id, metadata_row in met_df.iterrows(): 
            cell_idx+=1
            cell_file = os.path.join(slab_csv_dir,"{}".format(cell_id.replace(".swc",".csv")))
            if not os.path.exists(cell_file):
                print("could not find file for",met_type, cell_id)
                continue

            sp_df = pd.read_csv(cell_file)
            sp_df = sp_df[(~sp_df['slab_0'].isnull()) & (~sp_df['slab_2'].isnull())].copy()
            sp_df[['slab_0','slab_2']] = sp_df[['slab_0','slab_2']].astype(int)
            
            # inds = [i for i in sp_df[['slab_0','slab_2']].astype(int).values]
            this_met_image[sp_df.slab_2.values, sp_df.slab_0.values, cell_idx] = 1

            this_met_soma_locs.append(metadata_row.ccf_soma_location_nolayer)
        
        # get the histogram aggregates for the following soma locations
        soma_locations = ['VISp', 'AllHVAs', 'AllCells']
        for soma_loc in soma_locations:
            if soma_loc == "AllHVAs":
                specimen_slice_indices = [i for i, val in enumerate(this_met_soma_locs) if val != "VISp"]
                
            elif soma_loc=='AllCells':
                specimen_slice_indices = [i for i, val in enumerate(this_met_soma_locs) if True]
                
            else:
                specimen_slice_indices = [i for i, val in enumerate(this_met_soma_locs) if val==soma_loc]
        
        
            if specimen_slice_indices:

                this_met_specimen_subimage = this_met_image[:, :, specimen_slice_indices]
                
                contra_cells = []
                ipsi_cells = []
                
                # Iterate over each cell (z-slice)
                for z in range(this_met_specimen_subimage.shape[2]):
                    this_cell = this_met_specimen_subimage[:, :, z]
                    
                    ipsi = this_cell[:, 0:midline].sum(axis=1)
                    contra = this_cell[:, midline:].sum(axis=1)
                    
                    if normalize=='norm':
                        if contra.max() > 0:
                            contra = contra / contra.max()
                        if ipsi.max() > 0:
                            ipsi = ipsi / ipsi.max()
                            
                    elif normalize=='log':
                        contra = np.log1p(contra)
                        ipsi = np.log1p(ipsi)
                    
                    elif normalize != 'None':
                        raise ValueError
                        
                    contra_cells.append(contra)
                    ipsi_cells.append(ipsi)
                
                # Convert to arrays: shape (n_cells, x)
                contra_array = np.stack(contra_cells)
                ipsi_array = np.stack(ipsi_cells)
                
                # Mean and SEM
                contra_hist = contra_array.mean(axis=0)
                contra_sem = stats.sem(contra_array, axis=0)
                
                ipsi_hist = ipsi_array.mean(axis=0)
                ipsi_sem = stats.sem(ipsi_array, axis=0)

                
                met_type_slab_records[met_type][soma_loc] = {}
                met_type_slab_records[met_type][soma_loc]['ipsi_hist'] = ipsi_hist
                met_type_slab_records[met_type][soma_loc]['ipsi_sem'] = ipsi_sem
                met_type_slab_records[met_type][soma_loc]['contra_hist'] = contra_hist
                met_type_slab_records[met_type][soma_loc]['contra_sem'] = contra_sem

                print(f"{met_type}, {soma_loc}, n_contra_cells = {len(ipsi_cells)}")
                
                # print(f"adding {met_type}, {soma_loc}, total_n={full_met_df.shape[0]},contra_n1={met_df.shape[0]} contra_n2 = {len(ipsi_cells)}")
            else:
                True
                print("missing", met_type,soma_loc)
                # no data
        

        
    layer_tops = load_default_layer_template()
    layer_tops_normalized = {k:v/layer_tops['wm'] for k,v in layer_tops.items()}
    layer_tops_slab_dims = {k:v*top_shape[0] for k,v in layer_tops_normalized.items()}
        
    sps = """191813_6745-X2825-Y19481_reg.swc
    18453_5701-X28505-Y4297_reg.swc
    192343_7174-X3824-Y23571_reg.swc
    191812_6431-X3967-Y9193_reg.swc
    192343_7524-X5744-Y25593_reg.swc
    220307_6913-X8802-Y21284_reg.swc
    211541_7397-X6860-Y15587_reg.swc
    17109_6601-X5417-Y25287_reg.swc
    17109_7001-X6205-Y5194_reg.swc"""
    sp_ids = sps.split("\n")
    sp_ids = [s.replace(" ","") for s in sp_ids]






     
    
    fig = plt.figure(figsize= (6,5.5))
    gs  = GridSpec(nrows=len(sp_ids), ncols=3, width_ratios = [5, 0.05, 1,], wspace=0.05)
    first_hist_ax = None
    hist_axes = []
    for i, sp in enumerate(sp_ids):
        sp_file = os.path.join(slab_csv_dir, "{}".format(sp.replace(".swc",".csv")))
        
        sp_met_type = labels_df.loc[sp]['predicted_met_type']
        sp_soma_loc = labels_df.loc[sp]['ccf_soma_location_nolayer']
        

        hist_indexer = 'AllHVAs'
        if sp_soma_loc=='VISp':
            hist_indexer = 'VISp'
        
        col = color_dict[sp_met_type]
        
        sp_df = pd.read_csv(sp_file)
        morph_ax = fig.add_subplot(gs[i,0])

        if first_hist_ax is None:
            hist_ax = fig.add_subplot(gs[i, 2])
            first_hist_ax = hist_ax
        else:
            hist_ax = fig.add_subplot(gs[i, 2], sharex=first_hist_ax)
        
        # hist_ax = fig.add_subplot(gs[i,2])
        

        sampled_df = sp_df.iloc[::5]
        morph_ax.scatter(sampled_df['slab_0'], sampled_df['slab_2'], s=.1, edgecolors='none', c=col)
        morph_ax.set_xlim(0, top_shape[1])
        morph_ax.set_ylim(top_shape[0], 0)

        morph_ax.set(xticks=[], yticks=[])# anchor="SW", aspect='equal')
        
        print(sp, sp_met_type,sp_soma_loc)
        these_hist_data_dict = met_type_slab_records[sp_met_type][hist_indexer]
        mi = np.max(these_hist_data_dict['ipsi_hist'])
        mc = np.max(these_hist_data_dict['contra_hist'])
        print(f'ipsi max: {mi}\ncontra max:{mc}')
        
        plot_hist_to_axis(mean_hist = these_hist_data_dict['ipsi_hist'], 
                        sem_hist = these_hist_data_dict["ipsi_sem"],
                        ax=hist_ax,
                        negative=True,
                        color=col
                        )
        plot_hist_to_axis(mean_hist = these_hist_data_dict['contra_hist'], 
                        sem_hist = these_hist_data_dict["contra_sem"],
                        ax=hist_ax,
                        negative=False,
                        color=col
                        )

        for dec_ax in [morph_ax, hist_ax]:
            
            dec_ax.spines['top'].set_visible(False)
            dec_ax.spines['right'].set_visible(False)
            dec_ax.spines['left'].set_linewidth(0.5)
            dec_ax.spines['bottom'].set_linewidth(0.5)
            
            layer_lw = 0.5
            for v in layer_tops_slab_dims.values():
                dec_ax.axhline(v,zorder=-100,c='lightgrey',lw=layer_lw)
            dec_ax.axhline(0,zorder=-100,c='lightgrey',lw=layer_lw)

        morph_ax.axvline(main_shape[1]/2,linestyle='--',linewidth=0.5,c='k')
        hist_ax.axvline(0,linestyle='--',linewidth=0.5,c='k')

        
        hist_ax.set_yticks([])
        if normalize=='norm':
            hist_ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            if sp==sp_ids[-1]:
                hist_ax.set_xlabel('fraction of nodes',size=6)
        elif normalize=='log':
            hist_ax.set_xticks([-5,  0, 5])
            if sp==sp_ids[-1]:
                hist_ax.set_xlabel('log(# of nodes)',size=6)
        else:
            print('todo')
            
        hist_ax.tick_params(axis='x', width=0.5)
        if sp == sp_ids[-1]:
            hist_ax.set_xticklabels(hist_ax.get_xticklabels(),size=4)
            
        else:
            hist_ax.set_xticklabels(hist_ax.get_xticklabels(),size=0)
            hist_ax.tick_params(axis='x', length=2, width=0.3)
            # also change the x tick size
            # hist_ax.set_xticklabels([],size=6)

        morph_ax.set_ylim(top_shape[0], 0)
        
        hist_ax.set_ylim(top_shape[0], 0)
        hist_ax.set_ylabel(hist_indexer,fontsize=5)
        
        hist_axes.append(hist_ax)
    xlims = [ax.get_xlim() for ax in hist_axes]
    # try:
    #     assert all(x == xlims[0] for x in xlims)
    #     print("Not all x axis are equal on histotgrams")
    # except:
    #     del fig
        
    fig.savefig("plot_SuppFig_ContraCellTypes_MorphoSlab_AndHists.pdf",dpi=300,bbox_inches='tight')
    plt.clf()
    plt.close()
    
if __name__=="__main__":
    main()



