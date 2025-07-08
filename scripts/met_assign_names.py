import pandas as pd
import argschema as ags


MET_NAME_DICT = {
	0: "L2/3 IT",
	13: "L4 IT",
	6: "L4/L5 IT",
	11: "L5 IT-1",
	12: "L5 IT-2",
	16: "L5 IT-3 Pld5",
	8: "L6 IT-1",
	7: "L6 IT-2",
	15: "L6 IT-3",
	14: "L5/L6 IT Car3",
	10: "L5 ET-1 Chrna6",
	1: "L5 ET-2",
	2: "L5 ET-3",
	9: "L5 NP",
	3: "L6 CT-1",
	4: "L6 CT-2",
	5: "L6b",
}


class MetAssignNamesParameters(ags.ArgSchema):
    consensus_type_file = ags.fields.InputFile()
    met_name_file = ags.fields.OutputFile()


def main(args):
    met_df = pd.read_csv(args["consensus_type_file"], index_col=0)
    met_df["met_type"] = [MET_NAME_DICT[met] for met in met_df["met_type"]]
    met_df.to_csv(args["met_name_file"])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MetAssignNamesParameters)
    main(module.args)

