import argschema as ags
import pandas as pd


MET_TYPE_TO_SUBCLASS = {
	"L2/3 IT": "L2/3 IT",
	"L4 IT": "L4 & L5 IT",
	"L4/L5 IT": "L4 & L5 IT",
	"L5 IT-1": "L4 & L5 IT",
	"L5 IT-2": "L4 & L5 IT",
	"L5 IT-3 Pld5": "L4 & L5 IT",
	"L6 IT-1": "L6 IT",
	"L6 IT-2": "L6 IT",
	"L6 IT-3": "L6 IT",
	"L5/L6 IT Car3": "L6 IT Car3",
	"L5 ET-1 Chrna6": "L5 ET",
	"L5 ET-2": "L5 ET",
	"L5 ET-3": "L5 ET",
	"L5 NP": "L5 NP",
	"L6 CT-1": "L6 CT",
	"L6 CT-2": "L6 CT",
	"L6b": "L6b",
}

REF_SC_TO_SUBCLASS_MAP = {
    "L2/3 IT": "L2/3 IT",
    "L4": "L4 & L5 IT",
    "L5 IT": "L4 & L5 IT",
    "L6 IT": "L6 IT",
    "L5 PT": "L5 ET",
    "NP": "L5 NP",
    "L6 CT": "L6 CT",
    "L6b": "L6b",
}

class SubclassLabelsForPsParameters(ags.ArgSchema):
    inf_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met type text labels",
    )
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ref_tx_anno_file = ags.fields.InputFile()
    output_ps_file = ags.fields.OutputFile()
    output_ref_file = ags.fields.OutputFile()

def main(args):
    inf_met_type_df = pd.read_csv(args["inf_met_type_file"], index_col=0)
    ps_specimen_ids = inf_met_type_df.index.values

    ps_anno_df = pd.read_feather(args["ps_tx_anno_file"])
    ps_anno_df["spec_id_label"] = pd.to_numeric(ps_anno_df["spec_id_label"])
    ps_anno_df.set_index("spec_id_label", inplace=True)

    ps_sample_ids = ps_anno_df.loc[ps_specimen_ids, "sample_id"].values

    ps_data_df = pd.DataFrame(index=ps_sample_ids)
    ps_data_df.index.rename("sample_id", inplace=True)
    ps_data_df['dataset'] = 'patch-seq'
    ps_data_df['subclass'] = inf_met_type_df.loc[
        ps_specimen_ids, "inferred_met_type"].map(MET_TYPE_TO_SUBCLASS).values

    ps_ds_sc_labels = (ps_data_df['dataset'] + "-" + ps_data_df['subclass']).dropna()
    ps_ds_sc_labels.to_csv(args["output_ps_file"])

    ref_anno_df = pd.read_feather(args["ref_tx_anno_file"])
    ref_sample_ids = ref_anno_df["sample_id"].values
    ref_data_df = pd.DataFrame(index=ref_sample_ids)
    ref_data_df.index.rename("sample_id", inplace=True)
    ref_data_df['dataset'] = 'facs'
    ref_subclass = ref_anno_df.set_index("sample_id").loc[ref_data_df.index, "subclass_label"].map(REF_SC_TO_SUBCLASS_MAP)
    ref_subclass[ref_anno_df['cluster_label'].values == 'L6 IT VISp Car3'] = "L6 IT Car3"
    ref_data_df['subclass'] = ref_subclass.values

    ref_ds_sc_labels = (ref_data_df['dataset'] + "-" + ref_data_df['subclass']).dropna()
    ref_ds_sc_labels.to_csv(args["output_ref_file"])

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=SubclassLabelsForPsParameters)
    main(module.args)
