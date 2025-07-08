import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import argschema as ags
from tqdm import tqdm

class MetFlatmapDistTestParameters(ags.ArgSchema):
    inf_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met type text labels",
    )
    ccf_flat_coords_file = ags.fields.InputFile(
        description="csv file with ccf flatmap coordinates",
    )
    output_file = ags.fields.OutputFile()


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


def main(args):
    inf_met_type_df = pd.read_csv(args['inf_met_type_file'], index_col=0)
    ccf_flat_coords_df = pd.read_csv(args['ccf_flat_coords_file'], index_col=0)

    # Some cells belong to a visual area but are in a streamline that ends up in a non-visual area
    # We will not analyze those here
    ccf_flat_coords_df = ccf_flat_coords_df.loc[ccf_flat_coords_df['top_in_allowed_region'], :]

    # Fit the distribution of all sampled Patch-seq cells
    x = ccf_flat_coords_df['x'].values
    y = ccf_flat_coords_df['y'].values
    kernel = stats.gaussian_kde(np.vstack([x, y]))

    # Find actual log-likelihoods of each t-type & compare to random permutations

    n_resamples = 20000

    ccf_flat_coords_df['llik'] = np.log(kernel(np.vstack([x, y])))

    pvals = []
    for met_type in MET_TYPE_ORDER:
        fixed_met_type_name = met_type.replace("PT", "ET")
        print(fixed_met_type_name)

        specimen_ids = inf_met_type_df.index[inf_met_type_df['inferred_met_type'] == met_type]
        common_ids = ccf_flat_coords_df.index.intersection(specimen_ids)

        actual_llik = ccf_flat_coords_df.loc[common_ids, 'llik'].sum()

        n_points = len(common_ids)

        resample_llik = np.zeros(n_resamples)
        for i in tqdm(range(n_resamples)):
            my_resample = ccf_flat_coords_df.sample(n=n_points, replace=False)
            resample_llik[i] = my_resample['llik'].sum()

        pvals.append(np.sum(resample_llik <= actual_llik) / n_resamples)

    reject, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')

    for m, p, padj in zip(MET_TYPE_ORDER, pvals, pvals_adj):
        print(m, p, padj)

    df = pd.DataFrame({
        "pval": pvals,
        "pval_adj": pvals_adj,
    }, index=MET_TYPE_ORDER)
    df.to_csv(args['output_file'])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MetFlatmapDistTestParameters)
    main(module.args)
