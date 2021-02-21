from bsseval import evaluate
import os
import sys
import json
import argparse
import numpy
from collections import defaultdict
from essentia.standard import MonoLoader

bss_metric_names = ["SDR", "ISR", "SIR", "SAR"]


def eval_hpss(drum_estimates, drum_references, other_references, n_segs, seg_len):
    drum_bss = {}
    bss_results = numpy.zeros(dtype=numpy.float32, shape=(n_segs, 4, 1))

    n_seg = 0
    for track_prefix in perc_estimates[algo_name].keys():
        if n_seg >= n_segs:
            break
        cum_est_per_algo = numpy.zeros(dtype=numpy.float32, shape=(2, seg_len, 1))
        cum_ref_per_algo = numpy.zeros(dtype=numpy.float32, shape=(2, seg_len, 1))

        harm_ref = harm_references[track_prefix]
        harm_est = harm_estimates[algo_name][track_prefix]
        loaded_harm_ref = MonoLoader(filename=harm_ref)().reshape(seg_len, 1)
        loaded_harm_est = MonoLoader(filename=harm_est)().reshape(seg_len, 1)

        cum_est_per_algo[0] = loaded_harm_est
        cum_ref_per_algo[0] = loaded_harm_ref

        perc_ref = perc_references[track_prefix]
        perc_est = perc_estimates[algo_name][track_prefix]
        loaded_perc_ref = MonoLoader(filename=perc_ref)().reshape(seg_len, 1)
        loaded_perc_est = MonoLoader(filename=perc_est)().reshape(seg_len, 1)

        cum_est_per_algo[1] = loaded_perc_est
        cum_ref_per_algo[1] = loaded_perc_ref

        bss_metrics_segs = evaluate(cum_ref_per_algo, cum_est_per_algo)
        bss_metrics = numpy.nanmedian(bss_metrics_segs, axis=2)
        bss_results[n_seg][:] = numpy.asarray(bss_metrics)
        n_seg += 1

    total["drum_bss"][algo_name] = {}

    harm_bss = numpy.nanmedian(bss_results[:, :, 0], axis=0)

    for i, bss_metric_name in enumerate(bss_metric_names):
        total["harmonic_bss"][algo_name][bss_metric_name] = float(harm_bss[i])

    perc_bss = numpy.nanmedian(bss_results[:, :, 1], axis=0)

    for i, bss_metric_name in enumerate(bss_metric_names):
        total["percussive_bss"][algo_name][bss_metric_name] = float(perc_bss[i])

    return total


if __name__ == "__main__":
    args = parse_args()

    seg_len = args.seg_len_samples

    if not args.vocal:
        perc_estimates = defaultdict(dict)
        perc_references = defaultdict(dict)

        harm_estimates = defaultdict(dict)
        harm_references = defaultdict(dict)

        for song in os.scandir(hpss_results_dir):
            for dir_name, _, file_list in os.walk(song):
                algo_name = dir_name.split("/")[-1]
                if args.algorithms:
                    if algo_name not in args.algorithms.split(","):
                        continue
                for sep_file in file_list:
                    track_prefix = sep_file.split("_")[0]
                    if "percussive" in sep_file:
                        perc_estimates[algo_name][track_prefix] = os.path.join(
                            dir_name, sep_file
                        )
                    elif "harmonic" in sep_file:
                        harm_estimates[algo_name][track_prefix] = os.path.join(
                            dir_name, sep_file
                        )

        for dir_name, _, file_list in os.walk(hpss_data_dir):
            for ref_file in file_list:
                track_prefix = ref_file.split("_")[0]
                if "percussive" in ref_file:
                    perc_references[track_prefix] = os.path.join(dir_name, ref_file)
                elif "harmonic" in ref_file:
                    harm_references[track_prefix] = os.path.join(dir_name, ref_file)

        # percussive first
        n_segs = len(perc_references)

        bss_json = eval_hpss(
            harm_estimates,
            harm_references,
            perc_estimates,
            perc_references,
            n_segs,
            seg_len,
        )
        print(json.dumps(bss_json))

    # VOCAL case
    else:
        perc_estimates = defaultdict(dict)
        perc_references = defaultdict(dict)

        harm_estimates = defaultdict(dict)
        harm_references = defaultdict(dict)

        vocal_estimates = defaultdict(dict)
        vocal_references = defaultdict(dict)

        for song in os.scandir(vocal_results_dir):
            for dir_name, _, file_list in os.walk(song):
                algo_name = dir_name.split("/")[-1]
                if args.algorithms:
                    if algo_name not in args.algorithms.split(","):
                        continue
                for sep_file in file_list:
                    track_prefix = sep_file.split("_")[0]
                    if "percussive" in sep_file:
                        perc_estimates[algo_name][track_prefix] = os.path.join(
                            dir_name, sep_file
                        )
                    elif "harmonic" in sep_file:
                        harm_estimates[algo_name][track_prefix] = os.path.join(
                            dir_name, sep_file
                        )
                    elif "vocal" in sep_file:
                        vocal_estimates[algo_name][track_prefix] = os.path.join(
                            dir_name, sep_file
                        )

        for dir_name, _, file_list in os.walk(vocal_data_dir):
            for ref_file in file_list:
                track_prefix = ref_file.split("_")[0]
                if "percussive" in ref_file:
                    perc_references[track_prefix] = os.path.join(dir_name, ref_file)
                elif "harmonic" in ref_file:
                    harm_references[track_prefix] = os.path.join(dir_name, ref_file)
                elif "vocal" in ref_file:
                    vocal_references[track_prefix] = os.path.join(dir_name, ref_file)

        # percussive first
        n_segs = len(perc_references)

        bss_json = eval_vocal(
            harm_estimates,
            harm_references,
            perc_estimates,
            perc_references,
            vocal_estimates,
            vocal_references,
            n_segs,
            seg_len,
        )
        print(json.dumps(bss_json))
