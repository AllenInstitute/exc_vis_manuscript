import json
import h5py
from tqdm.contrib.concurrent import process_map
import argschema as ags
import numpy as np
import pandas as pd
import ipfx.script_utils as su


class CollectL6bSubthreshParameters(ags.ArgSchema):
    inf_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met type text labels",
    )
    ephys_sweep_info_file = ags.fields.InputFile(
        description="json file with ephys sweep info",
    )
    output_file = ags.fields.OutputFile(
        description="HDF5 file with traces",
    )

def get_trace(spec_info):
    extra_t = 0.1
    spec_id = spec_info["specimen_id"]
    n90_sweep_num = spec_info['subthresh_sweep_minus90']
    if n90_sweep_num is None:
        return {}

    start = spec_info['start']
    end = spec_info['end']
    data_set = su.dataset_for_specimen_id(spec_id,
        'lims-nwb2', None, None)
    swp = data_set.sweep(n90_sweep_num)
    v = swp.v
    t = swp.t
    start_ind = np.flatnonzero(t >= start - extra_t)[0]
    end_ind = np.flatnonzero(t >= end + extra_t)[0]
    v = v[start_ind:end_ind]
    t = t[start_ind:end_ind]
    t -= t[0]

    return {"specimen_id": spec_id, "v": v, "t": t}


def main(args):
    with open(args['ephys_sweep_info_file'], "r") as f:
        ephys_sweep_info = json.load(f)
    ephys_sweep_info_by_id = {int(i['specimen_id']): i for i in ephys_sweep_info}
    inf_met_type_df = pd.read_csv(args['inf_met_type_file'], index_col=0)

    met_type = ["L6b"]

    spec_ids = inf_met_type_df.loc[inf_met_type_df["inferred_met_type"].isin(met_type), :].index.values

    parallel_inputs = [ephys_sweep_info_by_id[i] for i in spec_ids if i in ephys_sweep_info_by_id]
    results = process_map(get_trace, parallel_inputs, max_workers=8, chunksize=5)
    results = [r for r in results if "specimen_id" in r]
    print(len(results))


    with h5py.File(args['output_file'], "w") as h5f:
        for r in results:
            g = h5f.create_group(str(r['specimen_id']))
            g.create_dataset('v', data=r['v'])
            g.create_dataset('t', data=r['t'])



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CollectL6bSubthreshParameters)
    main(module.args)

