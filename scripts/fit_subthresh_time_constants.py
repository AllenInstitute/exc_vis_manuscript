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


class FitSubthreshTimeConstantsParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(description="file with specimen IDs")
    manual_fail_sweep_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()
    lower_pct = ags.fields.Float(default=0.1)
    upper_pct = ags.fields.Float(default=0.9)


def exp_func(x, A, tau):
    return A * (1 - np.exp(-x / tau))


def fit_specimen(input_tuple):
    spec_id, ontology, data_source, lower_pct, upper_pct, manual_fail_sweep_dict = input_tuple

    data_set = su.dataset_for_specimen_id(spec_id, data_source, ontology, None)

    lsq_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
        ontology.long_square_names, sweep_qc_option="lims-passed-except-delta-vm",
        specimen_id=spec_id)

    if manual_fail_sweep_dict is not None and spec_id in manual_fail_sweep_dict:
        lsq_sweep_numbers = np.array([sn for sn in lsq_sweep_numbers if sn not in manual_fail_sweep_dict[spec_id]])

    (lsq_sweeps,
    lsq_features,
    lsq_an,
    lsq_start,
    lsq_end) = su.preprocess_long_square_sweeps(data_set, lsq_sweep_numbers)

    result_list = []

    for i in lsq_features['subthreshold_sweeps'].index:
        swp = lsq_sweeps.sweeps[i]
        baseline = lsq_features['subthreshold_sweeps'].at[i, 'v_baseline']
        stim_amp = int(np.round(lsq_features['subthreshold_sweeps'].at[i, 'stim_amp']))
        peak = lsq_features['subthreshold_sweeps'].at[i, 'peak_deflect'][0]
        delta = peak - baseline

        peak_ind = lsq_features['subthreshold_sweeps'].at[i, 'peak_deflect'][1]
        start_ind = tsu.find_time_index(swp.t, lsq_start)

        p0 = (1, 0.01)
        v_section = swp.v[start_ind:peak_ind] - baseline
        t_section = swp.t[start_ind:peak_ind]

        if delta < 0:
            rel_ind_lower = np.flatnonzero(v_section <= lower_pct * delta)[0]
            rel_ind_upper = np.flatnonzero(v_section <= upper_pct * delta)[0]
        else:
            rel_ind_lower = np.flatnonzero(v_section >= lower_pct * delta)[0]
            rel_ind_upper = np.flatnonzero(v_section >= upper_pct * delta)[0]

        pfit, _ = curve_fit(
            exp_func,
            t_section[rel_ind_lower:rel_ind_upper] - lsq_start,
            v_section[rel_ind_lower:rel_ind_upper],
            p0=p0
        )
        result_list.append(
            (spec_id, swp.sweep_number, stim_amp, pfit[1])
        )
    return result_list

def main():
    module = ags.ArgSchemaParser(schema_type=FitSubthreshTimeConstantsParameters)

    lower_pct = module.args['lower_pct']
    upper_pct = module.args['upper_pct']

    manual_fail_df = pd.read_csv(module.args['manual_fail_sweep_file'])
    manual_fail_sweep_dict = {}
    for specimen_id in manual_fail_df.specimen_id.unique():
        sweeps_for_specimen = manual_fail_df.loc[manual_fail_df.specimen_id == specimen_id, "sweep_number"].tolist()
        manual_fail_sweep_dict[specimen_id] = sweeps_for_specimen

    specimen_ids = np.loadtxt(module.args['input_file'], dtype=int)

    ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
    data_source = 'lims-nwb2'

    results = {}

    map_input = [(spec_id, ontology, data_source, lower_pct, upper_pct, manual_fail_sweep_dict)
        for spec_id in specimen_ids]

    results = process_map(fit_specimen, map_input, chunksize=1)

    results_flat = []
    for r in results:
        results_flat += r
    df = pd.DataFrame(results_flat, columns=["specimen_id", "sweep_number", "stim_amp", "tau"])
    df.to_csv(module.args['output_file'])


if __name__ == "__main__":
    main()