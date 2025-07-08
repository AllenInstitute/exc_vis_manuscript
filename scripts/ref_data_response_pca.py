import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argschema as ags
from sklearn.decomposition import PCA


class FigRespDataPcaParameters(ags.ArgSchema):
    ref_tx_anno_file = ags.fields.InputFile()
    ref_tx_data_file = ags.fields.InputFile()
    resp_gene_file = ags.fields.InputFile()
    n_pca_components = ags.fields.Integer(default=20)
    output_weights_file = ags.fields.OutputFile()
    output_means_file = ags.fields.OutputFile()
    output_pca_file = ags.fields.OutputFile()


def main(args):
    ref_anno_df = pd.read_feather(args["ref_tx_anno_file"])
    ref_data_df = pd.read_feather(args["ref_tx_data_file"])
    ref_data_df.set_index('sample_id', inplace=True)

    response_gene_list = np.genfromtxt(args["resp_gene_file"], dtype=str)
    exc_mask = ref_anno_df.class_label == "Glutamatergic"
    exc_sample_ids = ref_anno_df.loc[exc_mask, "sample_id"].values
    exc_resp_data_df = ref_data_df.loc[exc_sample_ids, response_gene_list]
    log_cpm_exc_resp_df = (exc_resp_data_df + 1).apply(np.log2)

    n_pcs = args["n_pca_components"]
    pca = PCA(n_components=n_pcs)
    transformed = pca.fit_transform(log_cpm_exc_resp_df.values)

    exc_resp_pc_df = pd.DataFrame(transformed, index=exc_sample_ids)
    exc_resp_pc_df.to_csv(args["output_pca_file"])

    pc_weights_df = pd.DataFrame(pca.components_.T, index=response_gene_list)
    pc_weights_df.to_csv(args["output_weights_file"])

    pc_means_df = pd.DataFrame({"mean": pca.mean_}, index=response_gene_list)
    pc_means_df.to_csv(args["output_means_file"])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigRespDataPcaParameters)
    main(module.args)
