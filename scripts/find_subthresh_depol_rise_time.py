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


class FitSubthreshRiseTimeParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(description="file with specimen IDs")
    output_file = ags.fields.OutputFile()
    subthresh_fit_file = ags.fields.InputFile()
    ephys_info_file = ags.fields.InputFile()


def fit_specimen(input_tuple):
    spec_id, ontology, data_source, ephys_info, subthresh_sweep_num = input_tuple

    data_set = su.dataset_for_specimen_id(spec_id, data_source, ontology, None)

    start = ephys_info['start']
    end = ephys_info['end']

    sweep = data_set.sweep(subthresh_sweep_num)
    start_idx = tsu.find_time_index(sweep.t, start)
    end_idx = tsu.find_time_index(sweep.t, end)

    # Finding baseline
    baseline_start_idx = tsu.find_time_index(sweep.t, start - 0.1)
    baseline_v = np.mean(sweep.v[baseline_start_idx:start_idx])

    # Finding steady-state
    pre_end_idx = tsu.find_time_index(sweep.t, end - 0.1) # 100 ms before end
    final_v = np.mean(sweep.v[pre_end_idx:end_idx])

    # Finding 10% and 90% times
    delta_v = final_v - baseline_v
    ind_10pct = np.flatnonzero(sweep.v[start_idx:end_idx] >= (baseline_v + 0.1 * delta_v))[0] + start_idx
    ind_90pct = np.flatnonzero(sweep.v[start_idx:end_idx] >= (baseline_v + 0.9 * delta_v))[0] + start_idx
    t_10pct = sweep.t[ind_10pct]
    t_90pct = sweep.t[ind_90pct]
    risetime_10pct_90pct = t_90pct - t_10pct


    return {
        "specimen_id": spec_id,
        "sweep_number": subthresh_sweep_num,
        "baseline_v": baseline_v,
        "final_v": final_v,
        "delta_v": delta_v,
        "t_10pct": t_10pct,
        "t_90pct": t_90pct,
        "risetime_10pct_90pct": risetime_10pct_90pct,
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
    module = ags.ArgSchemaParser(schema_type=FitSubthreshRiseTimeParameters)
    main(module.args)