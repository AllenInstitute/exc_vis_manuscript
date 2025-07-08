import json
import numpy as np
import pandas as pd
import argschema as ags
import ipfx.script_utils as su
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju
import ipfx.time_series_utils as tsu
import ipfx.stim_features as stf
import ipfx.data_set_features as dsf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from scipy.optimize import curve_fit


class FitSubthreshHumpParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(description="file with specimen IDs")
    output_file = ags.fields.OutputFile()
    subthresh_fit_file = ags.fields.InputFile()
    ephys_info_file = ags.fields.InputFile()


def fit_specimen(input_tuple):
    spec_id, ontology, data_source, ephys_info, subthresh_sweep_num = input_tuple

    data_set = su.dataset_for_specimen_id(spec_id, data_source, ontology, None)

    start = ephys_info['start']
    end = ephys_info['end']

    # Info about rheobase
    rheobase_latency = ephys_info['features']['supra_set_rheobase_latency']
    sweep = data_set.sweep(ephys_info['rheo_sweep'])
    latency_idx = tsu.find_time_index(sweep.t, rheobase_latency + start)
    rheobase_amplitude = sweep.i[latency_idx]

    # Finding "hump"
    sweep = data_set.sweep(subthresh_sweep_num)
    start_idx = tsu.find_time_index(sweep.t, start)
    end_idx = tsu.find_time_index(sweep.t, end)
    hump_idx = np.argmax(sweep.v[start_idx:end_idx]) + start_idx
    hump_t = sweep.t[hump_idx]
    hump_v = sweep.v[hump_idx]

    pre_end_idx = tsu.find_time_index(sweep.t, end - 0.1) # 100 ms before end
    final_v = np.mean(sweep.v[pre_end_idx:end_idx])
    subthresh_amplitude = sweep.i[hump_idx]

    return {
        "specimen_id": spec_id,
        "rheobase_amplitude": np.round(rheobase_amplitude),
        "rheobase_latency": rheobase_latency,
        "subthresh_amplitude": np.round(subthresh_amplitude),
        "hump_v": hump_v,
        "final_v": final_v,
        "hump_t": hump_t,
        "hump_latency": hump_t - start,
        "hump_amplitude": hump_v - final_v,
    }



def main(args):
    specimen_ids = np.loadtxt(args['input_file'], dtype=int)

    with open(args['ephys_info_file'], "r") as f:
        ephys_info = json.load(f)
    ephys_info_dict = {e['specimen_id']: e for e in ephys_info if e is not None}

    fit_df = pd.read_csv(args['subthresh_fit_file'], index_col=0)
    subthresh_dict = {}
    for sp_id in specimen_ids:
        depol_df = fit_df.loc[(fit_df['specimen_id'] == sp_id) & (fit_df['stim_amp'] > 0), :].sort_values("stim_amp", ascending=False)
        if len(depol_df['sweep_number'].values) == 0:
            continue
        subthresh_dict[sp_id] = int(depol_df['sweep_number'].values[0])

    ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
    data_source = 'lims-nwb2'

    map_input = [(spec_id, ontology, data_source, ephys_info_dict[spec_id], subthresh_dict[spec_id])
        for spec_id in specimen_ids if (spec_id in ephys_info_dict) and (spec_id in subthresh_dict)]

    results = process_map(fit_specimen, map_input, chunksize=1)

    df = pd.DataFrame.from_records(results).set_index("specimen_id")
    df.to_csv(args['output_file'])

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FitSubthreshHumpParameters)
    main(module.args)