#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
from fig_common import get_mesh_length_width_ratio

RUN_ALL=False # if True, re-run all experiments, including those with existing logfile data

def run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                    variable_cores, hbm_bw_cases=None, bw_per_core=None, mesh_calc=None):

    if variable_cores == True:
        assert bw_per_core is not None
    else:
        assert hbm_bw_cases is not None

    default_core_count = 5888
    if model in ["vit-huge"]:
        default_core_count = 1472

    for m in is_mesh:
        if m:
            assert mesh_calc is not None, "please specify a mesh function"
        for noc_bw in noc_bws:
            for batch_size in batch_sizes:
                for c in num_cores:
                    if not os.path.exists(f"benchmark_scripts/outputs/{model}/"):
                        os.makedirs(f"benchmark_scripts/outputs/{model}/")

                    for seq_length in seq_lengths:
                        output_dir = f"outputs_icbm_{seq_length}"
                        pickle_filename = f"{output_dir}/{c}cores/{model}-b{batch_size}/program.pickle"

                        if variable_cores == True:
                            hbm_bw_cases = [int(bw_per_core*c)]

                        for bw_case in hbm_bw_cases:
                            mesh_length_width_ratio = get_mesh_length_width_ratio(bw_case, noc_bw, c, mesh_calc, default_core_count) if m else 0.0
                            output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw_case}-{c}-{kb}-{mesh_length_width_ratio}-{noc_bw}"
                            timing_file = output_file + ".timing"
                            coldhot_file = output_file + ".coldhot"

                            print(output_file)

                            if not RUN_ALL:
                                if os.path.exists(output_file) and os.path.exists(timing_file) and os.path.exists(coldhot_file) and os.path.exists(pickle_filename):
                                    good = False
                                    for line in open(output_file, 'r'):
                                        if 'icbm_ms:' in line:
                                            good = True
                                    if good:
                                        print("Skipping good output file: ", output_file)
                                        continue
                                    else:
                                        print("Bad output file: ", output_file)

                            cmd = \
                                f"python3 icbm_launch.py {model} {c} {bw_case} --core_mem_kb {kb} --output_dir {output_dir} \
                                --layers {num_layers} --batch_size {batch_size} --sequence_length {seq_length} \
                                --use_pickle --launch_baseline --launch_icbm_full --reduce_order_list --output_cold_hot_list \
                                --split_factor {split_factor} --output_full_timings {timing_file} \
                                --comm {noc_bw} --mesh {mesh_length_width_ratio} > {output_file}"

                            print(cmd, end="... ", flush=True)
                            os.system(cmd)

def run_experiments_training(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, hbm_bw_cases, noc_bws, comp_bws,
                             is_mesh):
    for m in is_mesh: 
        for noc_bw in noc_bws:
            for comp_bw in comp_bws:
                for batch_size in batch_sizes:
                    for c in num_cores:
                        if not os.path.exists(f"benchmark_scripts/outputs/{model}/"):
                            os.makedirs(f"benchmark_scripts/outputs/{model}/")

                        for seq_length in seq_lengths:
                            output_dir = f"outputs_icbm_{seq_length}_training"
                            pickle_filename = f"{output_dir}/{c}cores/{model}-b{batch_size}/program.pickle"

                            for bw_case in hbm_bw_cases:
                                if m:
                                    mesh = float(500/bw_case*noc_bw)
                                    mesh = mesh**0.5
                                else:
                                    mesh = 0.0

                                output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw_case}-{c}-{kb}-{mesh}-{noc_bw}-{comp_bw}-training"
                                timing_file = output_file + ".timing"
                                coldhot_file = output_file + ".coldhot"

                                if not RUN_ALL:
                                    if os.path.exists(output_file) and os.path.exists(timing_file) and os.path.exists(coldhot_file) and os.path.exists(pickle_filename):
                                        good = False
                                        for line in open(output_file, 'r'):
                                            if 'icbm_ms:' in line:
                                                good = True
                                        if good:
                                            print("Skipping good output file: ", output_file)
                                            continue
                                        else:
                                            print("Bad output file: ", output_file)

                                cmd = \
                                    f"python3 icbm_launch.py {model} {c} {bw_case} --core_mem_kb {kb} --output_dir {output_dir} \
                                    --layers {num_layers} --batch_size {batch_size} --sequence_length {seq_length} \
                                    --use_pickle --launch_baseline --launch_icbm_full --reduce_order_list --output_cold_hot_list \
                                    --split_factor {split_factor} --output_full_timings {timing_file} \
                                    --comm {noc_bw} --mesh {mesh} --comp {comp_bw} --training > {output_file}" 

                                print(cmd, end="... ", flush=True)
                                os.system(cmd)


if __name__=="__main__":
    print("Generating data...")
    # there is overlap between the data needed for these figures, however, if the necessary data exists
    # it should be detected and skip re-running (so long as you don't change the top flags)

    # we explicitly include all experimental parameters and label them for completeness

    # fig. 17 (per-token serving latency)
    print("START: fig. 17 (per-token latency vs. HBM BW)")
    noc_bws = [1.0]
    batch_sizes = [16, 32, 64]
    num_cores = [5888]
    models = [
        (f"llama2-13", 40, 3),
        (f"llama2-70", 80, 3),
        (f"opt-30", 48, 4),
        (f"gemma2", 46, 2)
    ]
    kb = 624
    seq_lengths = [2048, 4096]
    hbm_bw_cases = [16000]
    is_mesh = [False]
    for model, num_layers, split_factor in models:
        run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                        variable_cores=False, hbm_bw_cases=hbm_bw_cases)
    print("END: fig. 17 (per-token latency vs. HBM BW)")

    # fig. 18 (execution breakdown)
    batch_sizes = [32]
    num_cores = [5888]
    kb = 624
    seq_lengths = [2048]
    models = [
        (f"llama2-13", 40, 3),
        (f"llama2-70", 80, 3),
        (f"opt-30", 48, 4),
        (f"gemma2", 46, 2)
    ]
    hbm_bw_cases = [16000]
    noc_bws = [1.0]
    is_mesh = [False]
    for model, num_layers, split_factor in models:
        run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                        variable_cores=False, hbm_bw_cases=hbm_bw_cases)

    # fig. 19 (per-token latency vs. HBM BW)
    print("START: fig. 19 (per-token latency vs. HBM BW)")
    noc_bws = [1.0]
    batch_sizes = [32]
    num_cores = [5888]
    models = [
        (f"llama2-13", 40, 3),
        (f"llama2-70", 80, 3),
        (f"opt-30", 48, 4),
        (f"gemma2", 46, 2)
    ]
    kb = 624
    seq_lengths = [2048]
    hbm_bw_cases = [i for i in range(4000, 16000+1, 1000)]
    is_mesh = [False, True]
    mesh_calc = "bw"
    for model, num_layers, split_factor in models:
        run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                        variable_cores=False, hbm_bw_cases=hbm_bw_cases, mesh_calc=mesh_calc)
    print("END: fig. 19 (per-token latency vs. HBM BW)")

    # fig. 20 (per-token latency breakdown vs. HBM BW)
    print("START: fig. 20 (per-token latency breakdown vs. HBM BW)")
    noc_bws = [1.0]
    batch_sizes = [32]
    num_cores = [5888]
    kb = 624
    seq_lengths = [2048]
    hbm_bw_cases = [i for i in range(6000, 16000+1, 1000)]
    is_mesh = [False]
    model, num_layers, split_factor = f"llama2-13", 40, 3
    run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                    variable_cores=False, hbm_bw_cases=hbm_bw_cases)
    print("END: fig. 20 (per-token latency breakdown vs. HBM BW)")

    # fig. 21 (interconnect utilization) - DONE
    print("START: fig. 21 (interconnect utilization")
    noc_bws = [1.0]
    batch_sizes = [32]
    num_cores = [5888]
    models = [
        (f"llama2-13", 40, 3),
        (f"llama2-70", 80, 3),
        (f"opt-30", 48, 4),
        (f"gemma2", 46, 2)
    ]
    kb = 624
    seq_lengths = [2048]
    hbm_bw_cases = [i for i in range(2000, 18000+1, 1000)]
    is_mesh = [False, True]
    mesh_calc = "bw"
    for model, num_layers, split_factor in models:
        run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                        variable_cores=False, hbm_bw_cases=hbm_bw_cases, mesh_calc=mesh_calc)
    print("END: fig. 21 (interconnect utilization")

    # fig. 22 (per-token latency vs. NoC BW)
    print("START: fig. 22 (per-token latency vs. NoC BW)")
    noc_bws = [0.75, 1.0, 1.25, 1.5]
    batch_sizes = [32]
    num_cores = [5888]
    kb = 624
    seq_lengths = [2048]
    hbm_bw_cases = [8000, 10000, 12000, 14000]
    is_mesh = [False, True]
    mesh_calc = "noc"
    model, num_layers, split_factor = "llama2-70", 80, 3
    run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                    variable_cores=False, hbm_bw_cases=hbm_bw_cases, mesh_calc=mesh_calc)
    print("END: fig. 22 (per-token latency vs. NoC BW)")

    # fig. 23 (per-token latency vs. core count)
    print("START: fig. 23 (per-token latency vs. core count)")
    # part 1. (large models)
    noc_bws = [1.0]
    batch_sizes = [32]
    num_cores = [(1472//2)*i for i in range(3, 8, 1)]
    models = [
        (f"llama2-13", 40, 3),
        (f"llama2-70", 80, 3),
        (f"opt-30", 48, 4),
        (f"gemma2", 46, 2)
    ]
    kb = 624
    seq_lengths = [2048]
    bw_per_core = 4000 / 1472
    is_mesh = [False, True]
    mesh_calc = "core"
    for model, num_layers, split_factor in models:
        run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                        variable_cores=True, bw_per_core=bw_per_core, mesh_calc=mesh_calc)
    # part 2. vit
    num_cores = [(1472//8)*i for i in range(3, 9, 1)]
    bw_per_core = 1000/1472
    model, num_layers, split_factor = "vit-huge", 32, 3
    run_experiments(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor, noc_bws, is_mesh,
                    variable_cores=True, bw_per_core=bw_per_core, mesh_calc=mesh_calc)
    print("END: fig. 23 (per-token latency vs. core count)")

    # fig. 24 (avg TFLOPS)
    noc_bws = [1.0, 1.5]
    comp_bws = [0.75, 1.0, 1.25, 1.5]
    batch_sizes = [2]
    num_cores = [5888]
    kb = 624
    seq_lengths = [2048]
    is_mesh = [False, True]
    hbm_bw_cases = [300, 400]
    model, num_layers, split_factor = "llama2-13", 40, 3
    run_experiments_training(batch_sizes, num_cores, kb, seq_lengths, model, num_layers, split_factor,
                             hbm_bw_cases, noc_bws, comp_bws, is_mesh)
