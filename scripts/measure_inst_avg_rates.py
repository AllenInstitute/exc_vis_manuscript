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


class InstAvgRatesParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(description="file with specimen IDs")
    ephys_info_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()

def fit_specimen(input_tuple):
    spec_id, ontology, data_source, ephys_info = input_tuple
    data_set = su.dataset_for_specimen_id(spec_id, data_source, ontology, None)
    sweeps_to_analyze = []
    for k in ("rheo_sweep", "plus30or40_sweep", "plus50or60_sweep"):
        if k in ephys_info and ephys_info[k] is not None:
            sweeps_to_analyze.append(ephys_info[k])


    sweeps = data_set.sweep_set(sweeps_to_analyze)
    start = ephys_info['start']
    end = ephys_info['end']
    spx, spfx = dsf.extractors_for_sweeps(
        sweeps,
        start=start,
        end=end,
        min_peak=-25,
        **dsf.detection_parameters(data_set.LONG_SQUARE)
    )

    results = []
    for swp in sweeps.sweeps:
        spikes_features = spx.process(swp.t, swp.v, swp.i)
        avg_rate = spikes_features.shape[0] / (end - start)
        if spikes_features.shape[0] > 1:
            isis = np.diff(spikes_features['threshold_t'].values)
            inst_freqs = 1 / isis
            max_inst_freq = inst_freqs.max()
        else:
            max_inst_freq = avg_rate
        results.append((spec_id, swp.sweep_number, avg_rate, max_inst_freq))
    return results


def main(args):
    specimen_ids = np.loadtxt(args['input_file'], dtype=int)

    with open(args['ephys_info_file'], "r") as f:
        ephys_info = json.load(f)
    ephys_info_dict = {e['specimen_id']: e for e in ephys_info if e is not None}

    ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
    data_source = 'lims-nwb2'

    map_input = [(spec_id, ontology, data_source, ephys_info_dict[spec_id])
        for spec_id in specimen_ids if spec_id in ephys_info_dict]

    results = process_map(fit_specimen, map_input, chunksize=1)
    results_flat = []
    for r in results:
        results_flat += r
    df = pd.DataFrame(results_flat, columns=("specimen_id", "sweep_number", "avg_rate", "max_inst_rate")).set_index("specimen_id")
    df.to_csv(args['output_file'])
    print(df)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=InstAvgRatesParameters)
    main(module.args)