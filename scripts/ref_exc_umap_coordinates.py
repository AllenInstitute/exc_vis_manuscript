import umap
import numpy as np
import pandas as pd
import argschema as ags
from sklearn.decomposition import PCA


class RefExcUmapCoordsParameters(ags.ArgSchema):
    ref_tx_anno_file = ags.fields.InputFile()
    ref_tx_data_file = ags.fields.InputFile()
    select_genes_file = ags.fields.InputFile()
    qc_file = ags.fields.InputFile()
    resp_gene_pc_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()

def main(args):
    # load data
    anno_df = pd.read_feather(args['ref_tx_anno_file'])
    genes_df = pd.read_csv(args['select_genes_file'], index_col=0)
    data_df = pd.read_feather(args['ref_tx_data_file'], columns=["sample_id"] + genes_df["Gene"].tolist()).set_index("sample_id")
    qc_df = pd.read_csv(args["qc_file"], index_col=0)
    resp_pc_df = pd.read_csv(args["resp_gene_pc_file"], index_col=0)

    # filter to excitatory cells
    anno_df = anno_df.loc[anno_df["class_label"] == "Glutamatergic", :]
    data_df = data_df.loc[anno_df["sample_id"], :]
    qc_df = qc_df.loc[data_df.index, :]

    print("data shape", data_df.shape)

    log_cpm_df = (data_df + 1).apply(np.log2)
    pca = PCA(n_components=20)
    transformed = pca.fit_transform(log_cpm_df.values)
    print(transformed.shape)
    drop_corr = 0.7

    corr = np.corrcoef(np.hstack([
        qc_df["log2Gene"].values.reshape(-1, 1),
        transformed
    ]), rowvar=False)
    pc_mask = np.abs(corr[0, :][1:]) < drop_corr
    print(f"qc filter: dropping {len(pc_mask) - pc_mask.sum()}")

    transformed = transformed[:, pc_mask]

    umap_coord = umap.UMAP(n_neighbors=25, min_dist=0.4).fit_transform(transformed)

    umap_df = pd.DataFrame(umap_coord, index=data_df.index, columns=['x', 'y'])
    umap_df['subclass_label'] = anno_df.set_index("sample_id").loc[umap_df.index, 'subclass_label']
    umap_df['subclass_color'] = anno_df.set_index("sample_id").loc[umap_df.index, 'subclass_color']
    umap_df['resp_pc_1'] = resp_pc_df.loc[umap_df.index, "0"].values
    print(umap_df.head())
    umap_df.to_csv(args['output_file'])






if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=RefExcUmapCoordsParameters)
    main(module.args)
