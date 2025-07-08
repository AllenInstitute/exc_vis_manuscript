import numpy as np
import pandas as pd
from ipfx.script_utils import dataset_for_specimen_id, categorize_iclamp_sweeps, preprocess_long_square_sweeps
from ipfx.stimulus import StimulusOntology
from ipfx.time_series_utils import average_voltage
import ipfx.lims_queries as lq
import allensdk.core.json_utilities as ju
from scipy.ndimage import gaussian_filter1d
import argschema as ags
import json as json
from multiprocessing import Pool
from functools import partial
import logging
import traceback
from scipy.optimize import curve_fit
from tqdm.contrib.concurrent import process_map


class SubthresholdFeaturesParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(
        description=("Input file of specimen IDs (one per line)"
            "- optional if LIMS is source"),
        default=None,
        allow_none=True
    )
    data_source = ags.fields.String(
        description="Source of NWB files ('sdk', 'lims', or 'lims-nwb2')",
        default="sdk",
        validate=lambda x: x in ["sdk", "lims", "lims-nwb2", "filesystem"]
        )
    output_file = ags.fields.OutputFile()


def evaluate_specimen(
    specimen_id,
    data_source,
    ontology,
    ):

    logging.info(f"Starting to process {specimen_id}")
    logging.debug("specimen_id: {}".format(specimen_id))
    data_set = dataset_for_specimen_id(specimen_id, data_source, ontology)
    if type(data_set) is dict and "error" in data_set:
        logging.warning("Problem getting data set for specimen {:d} from LIMS".format(specimen_id))
        return data_set

    try:
        lsq_sweep_numbers = categorize_iclamp_sweeps(
            data_set,
            ontology.long_square_names,
            sweep_qc_option="lims-passed-except-delta-vm",
            specimen_id=specimen_id
        )

        (lsq_sweeps,
        lsq_features,
        _,
        lsq_start,
        lsq_end) = preprocess_long_square_sweeps(data_set, lsq_sweep_numbers)


        keys_to_use = [
            'v_baseline',
            'sag',
            'vm_for_sag',
            'input_resistance',
            'tau',
        ]
        subthresh_features = {k: float(lsq_features[k]) for k in keys_to_use}

        subthresh_df = lsq_features["subthreshold_sweeps"]

        # calculate avg around peak & steady-state

        # Parameters for the averaging (matches sag calculation defaults)
        peak_width = 0.005
        baseline_interval = 0.03

        subthresh_deflect = []
        for swp_ind in subthresh_df.index:
            swp = lsq_sweeps.sweeps[swp_ind]
            sweep_number = swp.sweep_number

            peak_deflect_v, peak_deflect_ind = subthresh_df.at[swp_ind, "peak_deflect"]

            # average around the peak
            v_peak_avg = average_voltage(swp.v, swp.t,
                start=swp.t[peak_deflect_ind] - peak_width / 2,
                end=swp.t[peak_deflect_ind] + peak_width / 2)

            # steady-state at end
            v_steady = average_voltage(swp.v, swp.t, start=lsq_end - baseline_interval,
                end=lsq_end)
            subthresh_deflect.append({
                "sweep_number": int(sweep_number),
                "stimulus_amplitude": float(subthresh_df.at[swp_ind, "stim_amp"]),
                "peak_deflect_v": float(peak_deflect_v),
                "avg_peak_deflect_v": float(v_peak_avg),
                "steady_v": float(v_steady),
            })

        logging.info(f"Successfully processed {specimen_id}")
        subthresh_features["subthresh_deflect"] = subthresh_deflect
        return subthresh_features
    except Exception as detail:
        logging.warning("Exception when analyzing specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}}


def main(
    input_file,
    data_source,
    output_file,
    **kwargs):

    specimen_ids = [int(s) for s in np.loadtxt(input_file, dtype=int)]

    ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))
    evaluate_partial = partial(evaluate_specimen,
                               data_source=data_source,
                               ontology=ontology)
    results = process_map(evaluate_partial, specimen_ids, max_workers=6, chunksize=5)

    with open(output_file, "w") as f:
        json.dump(dict(zip(specimen_ids, results)), f)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=SubthresholdFeaturesParameters)
    main(**module.args)