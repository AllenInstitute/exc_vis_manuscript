import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import seaborn as sns
import os


def main():

    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)
    
    fmost_metadata_file = args['fmost_metadata_file'] 
    color_file = args['color_file'] 
    stacked_pseq_wnm_umap_file = args['fmost_pseq_combined_dend_umap_coords'] 
    pseq_met_labels_file = args['pseq_met_labels_file'] 
    fmost_only_umap_file =  args["fmost_only_morpho_umap_embeddings"] 
    
    pseq_met_label_df = pd.read_csv(pseq_met_labels_file,index_col=0)
    pseq_met_dict = {str(k):v for k,v in pseq_met_label_df['met_type'].to_dict().items()}
    
    fmost_meta_df = pd.read_csv(fmost_metadata_file,index_col=0)
    fmost_met_dict = fmost_meta_df['predicted_met_type'].to_dict()
    
    combined_met_dict = {**pseq_met_dict, **fmost_met_dict}

    with open(color_file,"r") as f:
        color_dict = json.load(f)
        
    outdir = "./"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        

    stacked_df = pd.read_csv(stacked_pseq_wnm_umap_file,index_col=0)
    stacked_df['predicted_met_type'] = stacked_df.index.map(combined_met_dict)
    print(stacked_df['predicted_met_type'].value_counts())
    
    stacked_df["color"] = stacked_df['predicted_met_type'].map(color_dict)


    stacked_df_ivscc = stacked_df[stacked_df['dataset']=='ivscc']
    stacked_df_fmost = stacked_df[stacked_df['dataset']=='fMOST']

    sorted_pmets = sorted(stacked_df.predicted_met_type.unique(), key = lambda x: ( [sc in x for sc in ["IT","ET", "NP","CT",'L6b']].index(True),x))
    assert [] == sorted([s for s in stacked_df.predicted_met_type.unique() if s not in sorted_pmets])

    handle_list = []
    legend_dot_size=6
    fmost_only = []# ["L6b","ET-MET-1*"]
    for pmet in sorted_pmets:
        mtdf = stacked_df[stacked_df['predicted_met_type']==pmet]
        c = mtdf['color'].values[0]
        pmet_rename = pmet.replace("PT-MET","ET-MET")
        edgecolor=c
        if pmet in fmost_only:
            edgecolor = "k"
        
        pt = Line2D([0], [0], label=pmet_rename, marker='o', markersize=legend_dot_size, 
            markeredgecolor=edgecolor, markerfacecolor=c, linestyle='')
        handle_list.append(pt)
        
    pt = Line2D([0], [0], label='fMOST', marker='o', markersize=legend_dot_size, 
            markeredgecolor="k", markerfacecolor="white", linestyle='')
    handle_list.append(pt)



    configs = ['IT', "Non_IT", "AllCells"]
    for config in configs:

        fig,axe=plt.gcf(),plt.gca()
        dot_size=50
        axe.scatter(stacked_df_ivscc['combined_umap_embedding_1'],
                stacked_df_ivscc['combined_umap_embedding_2'],
                c='lightgrey', # FORMAT CONSISTENT WITH NG
            s=dot_size)
        axe.scatter(stacked_df_fmost['combined_umap_embedding_1'],
                stacked_df_fmost['combined_umap_embedding_2'],
                c='lightgrey',
            s=dot_size*0.75)
        
        if config == 'IT':
            this_stacked_df_ivscc = stacked_df_ivscc[stacked_df_ivscc['predicted_met_type'].str.contains("IT")]
            this_stacked_df_fmost = stacked_df_fmost[stacked_df_fmost['predicted_met_type'].str.contains("IT")]
        
        elif config=='Non_IT':
            this_stacked_df_ivscc = stacked_df_ivscc[~stacked_df_ivscc['predicted_met_type'].str.contains("IT")]
            this_stacked_df_fmost = stacked_df_fmost[~stacked_df_fmost['predicted_met_type'].str.contains("IT")]

        elif config == 'AllCells':
            this_stacked_df_ivscc = stacked_df_ivscc.copy() 
            this_stacked_df_fmost = stacked_df_fmost.copy()
        else:
            raise ValueError
            
        
        axe.scatter(this_stacked_df_ivscc['combined_umap_embedding_1'],
            this_stacked_df_ivscc['combined_umap_embedding_2'],
            c=this_stacked_df_ivscc['color'],
        s=dot_size)

        axe.scatter(this_stacked_df_fmost['combined_umap_embedding_1'],
            this_stacked_df_fmost['combined_umap_embedding_2'],
            c=this_stacked_df_fmost['color'],
        edgecolor='k',
        s=dot_size)
        
        seen_mets = set(this_stacked_df_ivscc.predicted_met_type.values).union(set(this_stacked_df_ivscc.predicted_met_type.values))
        seen_mets.add('fMOST')
        these_handles = [h for h in handle_list if h.get_label() in seen_mets]
        
        axe.legend(handles=these_handles,bbox_to_anchor=(1,0.5),loc='center left',prop={'size':8},title='MET-Type')

            

            # axe.set_aspect('equal')
        axe.set_xlabel("M-UMAP 1")
        axe.set_ylabel("M-UMAP 2")
        axe.set_title(f"fMOST and IVSCC Dendritic Feature UMAP ({config})")
        axe.spines['right'].set_visible(False)
        axe.spines['top'].set_visible(False)
        axe.set_xticks([])
        axe.set_yticks([])

        fig.set_size_inches(8,8)
        sns.despine()

        # ofile = os.path.join(outdir,f'plot_morph_umap_Pseq_fMOST_Combined_{config}.png')
        # fig.savefig(ofile,dpi=600,bbox_inches='tight')
        ofile = os.path.join(outdir,f'plot_morph_umap_Pseq_fMOST_Combined_{config}.pdf')
        fig.savefig(ofile,dpi=600,bbox_inches='tight')

        plt.clf()
        plt.close()
        
        
    #
    #
    #    Fmost only, local + complete axon UMAPs
    #  
    #

    
    
    fmost_umap_df = pd.read_csv(fmost_only_umap_file,index_col=0)
    fmost_umap_df = fmost_umap_df.merge(fmost_meta_df,left_index=True,right_index=True)
    
    fmost_umap_df["color"] = fmost_umap_df['predicted_met_type'].map(color_dict)


    configs = [
        {
            "data":"IT",
            "feature_list": "local_and_long_axon",
            "embedding_cols":['local_and_complete_axon_only_umap_embedding_1', 'local_and_complete_axon_only_umap_embedding_2'],
        },
            
        {
            "data":"Non_IT",
            "feature_list": "local_and_long_axon",
            "embedding_cols":['local_and_complete_axon_only_umap_embedding_1', 'local_and_complete_axon_only_umap_embedding_2'],
        },
        

        {
            "data":"IT",
            "feature_list": "local_axon",
            "embedding_cols":['local_axon_only_umap_embedding_1', 'local_axon_only_umap_embedding_2'],
        },
        
        {
            "data":"Non_IT",
            "feature_list": "local_axon",
            "embedding_cols":['local_axon_only_umap_embedding_1', 'local_axon_only_umap_embedding_2'],
        },
        
        {
            "data":"All",
            "feature_list": "local_axon",
            "embedding_cols":['local_axon_only_umap_embedding_1', 'local_axon_only_umap_embedding_2'],
        },

        
    ]


    for cfig_dict in configs:
        
        data = cfig_dict['data']
        feat_cfig = cfig_dict['feature_list']
        embedding_cols = cfig_dict['embedding_cols']
        
        if cfig_dict['data']=='IT':
            this_fmost_data = fmost_umap_df[fmost_umap_df['predicted_met_type'].str.contains("IT")]
        elif data=='Non_IT':
            this_fmost_data = fmost_umap_df[~fmost_umap_df['predicted_met_type'].str.contains("IT")]
        elif data == 'All':
            this_fmost_data = fmost_umap_df.copy()
        else:
            raise ValueError
        
        fig,axe=plt.gcf(),plt.gca()
        dot_size=50
        axe.scatter(fmost_umap_df[embedding_cols[0]],
                fmost_umap_df[embedding_cols[1]],
                c='lightgrey', # FORMAT CONSISTENT WITH NG
            s=dot_size)
        
        axe.scatter(this_fmost_data[embedding_cols[0]],
                this_fmost_data[embedding_cols[1]],
                c=this_fmost_data['color'], # FORMAT CONSISTENT WITH NG
            s=dot_size)
        
        
        seen_mets = set(this_fmost_data.predicted_met_type.values)
        these_handles = [h for h in handle_list if h.get_label() in seen_mets]
        
        axe.legend(handles=these_handles,bbox_to_anchor=(1,0.5),loc='center left',prop={'size':8},title='MET-Type')

        axe.set_xlabel("M-UMAP 1")
        axe.set_ylabel("M-UMAP 2")
        axe.set_title(f"fMOST Only {data} {feat_cfig} UMAP")
        axe.spines['right'].set_visible(False)
        axe.spines['top'].set_visible(False)
        axe.set_xticks([])
        axe.set_yticks([])

        fig.set_size_inches(8,8)
        sns.despine()

        # ofile = os.path.join(outdir,f'plot_morph_umap_fmost_{data}_{feat_cfig}.png')
        # fig.savefig(ofile,dpi=600,bbox_inches='tight')
        ofile = os.path.join(outdir,f'plot_morph_umap_fmost_{data}_{feat_cfig}.pdf')
        fig.savefig(ofile,dpi=600,bbox_inches='tight')

        # plt.show()
        plt.clf()
        plt.close()
        


        

if __name__ == '__main__':
    main()
