import json
import h5py
from tqdm.contrib.concurrent import process_map
import argschema as ags
import numpy as np
import pandas as pd
import ipfx.script_utils as su
from ipfx.feature_extractor import SpikeFeatureExtractor


class CollectRheoApWaveformParameters(ags.ArgSchema):
    ephys_sweep_info_file = ags.fields.InputFile(
        description="json file with ephys sweep info",
    )
    subset_id_file = ags.fields.InputFile(
        default=None, allow_none=True)
    output_file = ags.fields.OutputFile(
        description="HDF5 file with traces",
    )


def get_trace(spec_info):
    spec_id = spec_info["specimen_id"]
    rheo_sweep_num = spec_info['rheo_sweep']
    if rheo_sweep_num is None:
        return {"error": f"{spec_id}: None for rheo_sweep_num"}

    start = spec_info['start']
    end = spec_info['end']
    data_set = su.dataset_for_specimen_id(spec_id,
        'lims-nwb2', None, None)
    swp = data_set.sweep(rheo_sweep_num)

    step_size = 1
    extra_interval = 200

    # Check sampling rate - it's either 50kHz or 200 kHz, so use 100kHz as cutoff
    if swp.t[1] - swp.t[0] < 1/100000:
        # 200 kHz sampling
        step_size = 4
        extra_interval *= step_size

    sfx = SpikeFeatureExtractor(start=start, end=end, min_peak=-25)

    v = swp.v
    t = swp.t
    # handle NaNs in voltage trace

    nan_mask = np.isnan(v)
    v = v[~nan_mask]
    t = t[~nan_mask]
    spike_info = sfx.process(t, v, swp.i)

    if spike_info.shape[0] < 1:
        return {"error": f"{spec_id}: no spikes detected in sweep {rheo_sweep_num}"}

    thresh_index = int(spike_info["threshold_index"].values[0])
    peak_index = int(spike_info["peak_index"].values[0])

    ap_v = swp.v[peak_index - extra_interval:peak_index + extra_interval:step_size]
    ap_t = swp.t[peak_index - extra_interval:peak_index + extra_interval:step_size]
    thresh_delta = (peak_index - thresh_index) // step_size

    return {
        "specimen_id": spec_id,
        "ap_v": ap_v,
        "ap_t": ap_t,
        "thresh_delta": thresh_delta,
    }


def main(args):
    with open(args['ephys_sweep_info_file'], "r") as f:
        ephys_sweep_info = json.load(f)

    if args["subset_id_file"] is not None:
        subset_ids = np.loadtxt(args["subset_id_file"])
        ephys_sweep_info = [k for k in ephys_sweep_info if k["specimen_id"] in subset_ids]

    results = process_map(get_trace, ephys_sweep_info, max_workers=6, chunksize=5)
    errors = [r for r in results if "error" in r]
    with open("collect_rheo_ap_waveforms_errors.json", "w") as f:
        json.dump(errors, f, indent=True)

    results = [r for r in results if "specimen_id" in r]

    print(f"Processed {len(results)} cells")

    # Put together data structures
    ap_v = np.vstack([r["ap_v"] for r in results])
    ap_t = np.vstack([r["ap_t"] for r in results])
    specimen_ids = np.array([r["specimen_id"] for r in results])
    thresh_deltas = np.array([r["thresh_delta"] for r in results])

    with h5py.File(args['output_file'], "w") as h5f:
        h5f.create_dataset("ap_v", data=ap_v)
        h5f.create_dataset("ap_t", data=ap_t)
        h5f.create_dataset("specimen_id", data=specimen_ids)
        h5f.create_dataset("thresh_delta", data=thresh_deltas)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CollectRheoApWaveformParameters)
    main(module.args)

