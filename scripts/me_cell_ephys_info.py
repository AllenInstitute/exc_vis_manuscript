import numpy as np
import pandas as pd
import argschema as ags
import scipy.spatial.distance as distance
from scipy import stats
import json
import os
import mouse_met_figs.utils as utils
import allensdk.core.json_utilities as ju
from ipfx.stimulus import StimulusOntology
import ipfx.script_utils as su
from ipfx.aibs_data_set import AibsDataSet
import ipfx.feature_vectors as fv
from multiprocessing import Pool
import sklearn.utils as skutils


class MECellEphysInfoParameters(ags.ArgSchema):
    id_file = ags.fields.InputFile(
        description="text file with data set specimen IDs")
    manual_fail_sweep_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile(
        description="output json file name")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def example_sweeps(specimen_id, ontology, manual_fail_sweeps):
    data_set = su.dataset_for_specimen_id(specimen_id, data_source="lims-nwb2", ontology=ontology)
    if type(data_set) is dict and "error" in data_set:
#         print(data_set)
        return None

    print(specimen_id, type(specimen_id))
    lsq_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
        ontology.long_square_names, sweep_qc_option="lims-passed-except-delta-vm",
        specimen_id=specimen_id)

    if manual_fail_sweeps is not None and specimen_id in manual_fail_sweeps:
        lsq_sweep_numbers = np.array([sn for sn in lsq_sweep_numbers if sn not in manual_fail_sweeps[specimen_id]])

    try:
        (lsq_sweeps,
        lsq_features,
        lsq_an,
        lsq_start,
        lsq_end) = su.preprocess_long_square_sweeps(data_set, lsq_sweep_numbers)
    except Exception as detail:
        print("Exception when preprocessing long square sweeps from specimen {:d}".format(specimen_id))
        print(detail)
        return None

    lsq_sweep_number_lookup = {swp.sweep_number: i for i, swp in enumerate(lsq_sweeps.sweeps)}

    fx_rheo_sweep_num = int(lsq_sweeps.sweeps[lsq_features["rheobase_sweep"].name].sweep_number)
    fx_rheo_latency = lsq_features["rheobase_sweep"]["latency"]

    subthresh_df = lsq_features["subthreshold_membrane_property_sweeps"]
    subthresh_df.drop_duplicates(subset=["stim_amp"], inplace=True)

    subthresh_df_target = subthresh_df.loc[subthresh_df["stim_amp"] == -70, :]
    if subthresh_df_target.shape[0] < 1:
        subthresh_sweep_minus70_num = None
        sag_minus70 = None
    else:
        subthresh_sweep_minus70_num = int(lsq_sweeps.sweeps[subthresh_df_target.index.values[0]].sweep_number)
        sag_minus70 = subthresh_df_target["sag"].values[0]

    subthresh_df_target = subthresh_df.loc[subthresh_df["stim_amp"] == -90, :]
    if subthresh_df_target.shape[0] < 1:
        subthresh_sweep_minus90_num = None
        sag_minus90 = None
    else:
        subthresh_sweep_minus90_num = int(lsq_sweeps.sweeps[subthresh_df_target.index.values[0]].sweep_number)
        sag_minus90 = subthresh_df_target["sag"].values[0]

    target_amplitudes = np.arange(0, 100, 10)
    try:
        supra_sweep_list = fv.identify_suprathreshold_sweeps(
            lsq_sweeps, lsq_features, target_amplitudes, shift=10, amp_tolerance=4, needed_amplitudes=None)
    except Exception as detail:
        print("Exception when finding long square sweeps of right amplitude for {:d}".format(specimen_id))
        print(detail)
        return None
    rheo_sweep = supra_sweep_list[0]
    rheo_sweep_num = int(rheo_sweep.sweep_number)

    supra_set_rheo_latency = lsq_features["spiking_sweeps"].at[lsq_sweep_number_lookup[rheo_sweep_num], "latency"]

    latency = max(fx_rheo_latency, supra_set_rheo_latency)


    if (supra_sweep_list[4] is not None) or (supra_sweep_list[3] is not None):
        if supra_sweep_list[4] is not None:
            plus30or40_sweep_num = int(supra_sweep_list[4].sweep_number)
        else:
            plus30or40_sweep_num = int(supra_sweep_list[3].sweep_number)

        plus30or40_isi_cv = lsq_features["spiking_sweeps"].at[lsq_sweep_number_lookup[plus30or40_sweep_num], "isi_cv"]

        spike_data = lsq_features["spikes_set"][lsq_sweep_number_lookup[plus30or40_sweep_num]]
        thresh_t = spike_data["threshold_t"]
        spike_count = np.ones_like(thresh_t)
        one_ms = 0.001
        width = 20 # in ms
        duration = lsq_end - lsq_start
        n_bins = int(duration / one_ms) // width
        bin_edges = np.linspace(lsq_start, lsq_end, n_bins + 1) # includes right edge, so adding one to desired bin number
        output = stats.binned_statistic(thresh_t,
                                spike_count,
                                statistic='sum',
                                bins=bin_edges)[0]
        binary_output = output > 0
        plus30or40_spiking_fraction = float(np.sum(binary_output) / len(binary_output))
    else:
        plus30or40_sweep_num = None
        plus30or40_isi_cv = None
        plus30or40_spiking_fraction = None

    if (supra_sweep_list[6] is not None) or (supra_sweep_list[5] is not None):
        if supra_sweep_list[6] is not None:
            plus50or60_sweep_num = int(supra_sweep_list[6].sweep_number)
        else:
            plus50or60_sweep_num = int(supra_sweep_list[5].sweep_number)
        plus50or60_isi_cv = lsq_features["spiking_sweeps"].at[lsq_sweep_number_lookup[plus50or60_sweep_num], "isi_cv"]
    else:
        plus50or60_sweep_num = None
        plus50or60_isi_cv = None

    return {
        "specimen_id": specimen_id,
        "subthresh_sweep_minus70": subthresh_sweep_minus70_num,
        "subthresh_sweep_minus90": subthresh_sweep_minus90_num,
        "fx_rheo_sweep": fx_rheo_sweep_num,
        "rheo_sweep": rheo_sweep_num,
        "plus30or40_sweep": plus30or40_sweep_num,
        "plus50or60_sweep": plus50or60_sweep_num,
        "start": lsq_start,
        "end": lsq_end,
        "features": {
            "input_resistance": lsq_features["input_resistance"],
            "sag_minus70": sag_minus70,
            "sag_minus90": sag_minus90,
            "sag_fx": lsq_features["sag"],
            "vm_for_sag": lsq_features["vm_for_sag"],
            "fx_rheobase_latency": fx_rheo_latency,
            "supra_set_rheobase_latency": supra_set_rheo_latency,
            "plus30or40_isi_cv": plus30or40_isi_cv,
            "plus30or40_spiking_fraction": plus30or40_spiking_fraction,
            "plus50or60_isi_cv": plus50or60_isi_cv,
        }
    }


def main(id_file, manual_fail_sweep_file, output_file, **kwargs):
    dataset_ids = np.loadtxt(id_file, dtype=int)

    ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))


    manual_fail_df = pd.read_csv(manual_fail_sweep_file)
    manual_fail_sweep_dict = {}
    for specimen_id in manual_fail_df.specimen_id.unique():
        sweeps_for_specimen = manual_fail_df.loc[manual_fail_df.specimen_id == specimen_id, "sweep_number"].tolist()
        manual_fail_sweep_dict[specimen_id] = sweeps_for_specimen

#     example_sweeps(int(dataset_ids[0]), ontology)
#     return

    p = Pool()
    pool_input = [(int(sid), ontology, manual_fail_sweep_dict) for sid in dataset_ids]
    results = p.starmap(example_sweeps, pool_input)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MECellEphysInfoParameters)
    main(**module.args)
