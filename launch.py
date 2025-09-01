import argparse
import time
import os
import sys
import math
import pickle
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
from copy import deepcopy

from DNNModels import get_model_from_file, \
    search_optimal_exe_load_config_order_independent_helper, \
    search_optimal_exe_load_config_baseline_independent_helper
import DNNProgram

import TensorExpression as TE

from benchmark_scripts.fig_common import IPU_Mk2_cycle_to_ms

MAX_EDIT_DIST = 4

CORE_REDUCE = 3

TRAINING = False

def gen_pickle(args, output_dir, layer, order_pickle_filename, pickle_filename, log_file):
    print("Generating new pickle...", file=sys.stderr, flush=True)

    if TRAINING:
        cmd = f"cd models && python3.10 model_parser.py {args.modelname}.json {args.batch_size} {args.sequence_length} {args.num_cores // (args.split_factor+1)}"
        TE.MAX_MEM_THRESHOLD = 0.999
        TE.CORE_UTIL_THRESHOLD = 0.999
    else:
        cmd = f"cd models && python3.10 model_parser.py {args.modelname}.json {args.batch_size} 1 {args.num_cores // args.split_factor} {args.sequence_length}"
    os.system(cmd)

    prog = get_model_from_file(f"models/TExpr/TExpr_{args.modelname}-b{args.batch_size}.json", name=f"{args.modelname}-b{args.batch_size}",
                            output_dir=f"{output_dir}/{args.num_cores}cores",
                            num_cores=[args.num_cores], tot_mem_size_per_core=args.core_mem_kb*1024)

    start = time.perf_counter()

    prog.run_intra_op_optimization(num_threads=(int)(math.ceil((os.cpu_count() or 20)**0.6)))
    # prog.run_inter_op_optimization(num_threads=(int)(math.ceil((os.cpu_count() or 20)**0.6)), mem_size_threshold=mem_threshold)
    prog.generate_all_cold_hot_table(num_threads=(int)(math.ceil((os.cpu_count() or 20)**0.6)))
    prog.init_min_cold_size_bytes_per_core()

    if args.reduce_order_list:
        order_list = prog.init_all_order_lists_reduced(layer, max_edit_dist=MAX_EDIT_DIST)
    else:
        order_list = prog.init_all_order_lists(layer)

    with open(order_pickle_filename, 'wb') as f:
        pickle.dump(order_list, f)

    end = time.perf_counter()

    with open(pickle_filename, 'wb') as f:
        pickle.dump(prog, f)

    print(f"prepare time: {end - start} sec")
    log_file.write(f"prepare time: {end - start} sec\n")

    return prog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ICBM")
    parser.add_argument("modelname", type=str)
    parser.add_argument("num_cores", type=int)
    parser.add_argument("hbm_bw", type=int)
    parser.add_argument("--layers", required=True, type=int)

    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--sequence_length", required=True, type=int)

    parser.add_argument("--split_factor", required=True, type=int)

    parser.add_argument("--core_mem_kb", required=False, type=int, default=624)
    parser.add_argument("--output_dir", required=False, type=str, default="")

    parser.add_argument("--use_pickle", action='store_true')
    parser.add_argument("--launch_baseline", action='store_true')
    parser.add_argument("--launch_icbm_full", action='store_true')
    parser.add_argument("--update_order_list", action='store_true')
    parser.add_argument("--reduce_order_list", action='store_true')
    parser.add_argument("--report_all_baseline", action='store_true')

    parser.add_argument("--output_full_timings", required=False, default=None)

    parser.add_argument("--generate_pickle_only", action='store_true', required=False, default=False)
    parser.add_argument("--generate_order_only", action='store_true', required=False, default=False)
    parser.add_argument("--output_cold_hot_list", action='store_true', required=False, default=False)

    parser.add_argument("--mesh", required=False, type=float, default=0)
    parser.add_argument("--comm", required=False, type=float, default=1)
    parser.add_argument("--comp", required=False, type=float, default=1)
    parser.add_argument("--training", action='store_true', required=False, default=False)
    args = parser.parse_args()

    DNNProgram.MESH = args.mesh
    DNNProgram.COMM = args.comm
    DNNProgram.COMP = args.comp
    TRAINING = args.training
    DNNProgram.TRAINING = TRAINING
    if TRAINING:
        MAX_EDIT_DIST = 3

    if "vit" in args.modelname:
        CORE_REDUCE = max(4, CORE_REDUCE)

    # hbm_bw = args.hbm_bw # TODO I'd like to remove but I think the name is reused soemwhere
    # output_dir = args.output_dir # TODO why is this even a param
    if args.output_dir != "":
        output_dir = args.output_dir
    else:
        output_dir = f"out/outputs_icbm_{args.sequence_length}"
    layer = args.layers

    mem_threshold = TE.MAX_MEM_THRESHOLD

    # if TRAINING:
    #     output_dir = f"{output_dir}_training"

    if not os.path.exists(f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}-{args.hbm_bw}GBps/"):
        os.makedirs(f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}-{args.hbm_bw}GBps/")
    pickle_filename = f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}/program.pickle"
    order_pickle_filename = f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}/order.pickle"
    log_name= f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}-{args.hbm_bw}GBps.log"
    log_file = open(log_name, 'w')

    if not args.use_pickle or \
       not os.path.exists(f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}/program.pickle") or \
       not os.path.exists(order_pickle_filename):
        prog = gen_pickle(args, output_dir, layer, order_pickle_filename, pickle_filename, log_file)

    else:
        load = False
        try:
            with open(pickle_filename, 'rb') as f:
                prog:DNNProgram.DNNProgram = pickle.load(f)
                load = True
        except:
            print("Error loading pickle file", file=sys.stderr)
            prog = gen_pickle(args, output_dir, layer, order_pickle_filename, pickle_filename, log_file)

        if args.update_order_list and load:
            order_list = None
            if args.reduce_order_list:
                order_list = prog.init_all_order_lists_reduced(layer, max_edit_dist=MAX_EDIT_DIST)
            else:
                order_list = prog.init_all_order_lists(layer)

            with open(order_pickle_filename, 'wb') as f:
                pickle.dump(order_list, f)

            for op in prog.ops:
                op.expr.cold_config_candidates = {}
                op.expr.config_dict = {}
                for cold in op.expr.cold_hot_table:
                    for hot in op.expr.cold_hot_table[cold]:
                        op.expr.cold_hot_table[cold][hot][0][0] = ()
                        op.expr.cold_hot_table[cold][hot][1][0] = ()

            with open(pickle_filename, 'wb') as f:
                pickle.dump(prog, f)

    if args.generate_pickle_only or args.generate_order_only:
        exit(0)

    all_op_orderings = []
    with open(order_pickle_filename, 'rb') as f:
        all_op_orderings = deepcopy(pickle.load(f))

    print(f"num_orders: {len(all_op_orderings)}")
    log_file.write(f"num_orders: {len(all_op_orderings)}\n")

    if TRAINING:
        all_op_orderings = [list(range(len(prog.ops)))]

    timing_file = None
    if args.output_full_timings is not None:
        timing_file = open(args.output_full_timings, "w")

    benchmark_trace_name = f"benchmark_scripts/outputs/{args.modelname}/test-bw-{args.batch_size}-{args.sequence_length}-{args.hbm_bw}-{args.num_cores}-{args.core_mem_kb}-{args.mesh}-{args.comm}"
    if TRAINING:
        benchmark_trace_name = f"{benchmark_trace_name}-{args.comp}-training"

    baseline_file = None
    if args.report_all_baseline:
        baseline_file = open(f"{benchmark_trace_name}.baseline", "w")

    cold_hot_file = None
    if args.output_cold_hot_list:
        cold_hot_file = open(f"{benchmark_trace_name}.coldhot", "w")

    # naive
    start = time.perf_counter()
    naive_exe_list_1, naive_load_list_1, naive_cold_hot_list_1, naive_comp_shift_list_1, \
        naive_num_overlap_load_1, naive_load_remain_noc_list_1 = prog.naive_search_optimal_exe_load_config_all(args.hbm_bw, layer, use_largest_cold=False)

    naive_exe_list_2, naive_load_list_2, naive_cold_hot_list_2, naive_comp_shift_list_2, \
        naive_num_overlap_load_2, naive_load_remain_noc_list_2 = prog.naive_search_optimal_exe_load_config_all(args.hbm_bw, layer, use_largest_cold=True)

    if -naive_load_list_1[0][0] < -naive_load_list_2[0][0]:
        naive_exe_list, naive_load_list, naive_cold_hot_list, naive_comp_shift_list, \
            naive_num_overlap_load, naive_load_remain_noc_list = \
        naive_exe_list_1, naive_load_list_1, naive_cold_hot_list_1, naive_comp_shift_list_1, \
            naive_num_overlap_load_1, naive_load_remain_noc_list_1
    else:
        naive_exe_list, naive_load_list, naive_cold_hot_list, naive_comp_shift_list, \
            naive_num_overlap_load, naive_load_remain_noc_list = \
        naive_exe_list_2, naive_load_list_2, naive_cold_hot_list_2, naive_comp_shift_list_2, \
            naive_num_overlap_load_2, naive_load_remain_noc_list_2

    end = time.perf_counter()
    print(f"\nnaive schedule time: {end - start} sec")
    log_file.write(f"\nnaive schedule time: {end - start} sec\n")

    naive_exe_cycles = sum([exe[1]-exe[0] for exe in naive_exe_list])
    naive_load_cycles = sum([load[1]-load[0] for load in naive_load_list])
    naive_cycles = -naive_load_list[0][0]

    naive_exe_ms = IPU_Mk2_cycle_to_ms(naive_exe_cycles)
    naive_load_ms = IPU_Mk2_cycle_to_ms(naive_load_cycles)
    naive_ms = IPU_Mk2_cycle_to_ms(naive_cycles)

    print(f"naive_exe_ms: {naive_exe_ms}")
    print(f"naive_load_ms: {naive_load_ms}")
    print(f"naive_ms: {naive_ms}")
    log_file.write(f"naive_exe_ms: {naive_exe_ms}\n")
    log_file.write(f"naive_load_ms: {naive_load_ms}\n")
    log_file.write(f"naive_ms: {naive_ms}\n")

    if timing_file is not None:
        timing_file.write("Naive:\n")
        timing_file.write(str(naive_exe_list))
        timing_file.write("\n")
        timing_file.write(str(naive_load_list))
        timing_file.write("\n")
        timing_file.write(str(naive_comp_shift_list))
        timing_file.write("\n")
        timing_file.write(str(naive_load_remain_noc_list))
        timing_file.write("\n")

    if cold_hot_file is not None:
        cold_hot_file.write("Naive:\n")
        cold_hot_file.write(str(naive_cold_hot_list))
        cold_hot_file.write("\n")

    # baseline
    if args.launch_baseline:
        min_exec_byte = 0
        for op in prog.ops:
            min_cold_size = next(iter(op.expr.cold_hot_table))
            min_hot_size = next(iter(op.expr.cold_hot_table[min_cold_size]))
            min_exec_byte = max(min_exec_byte, min_hot_size)
        min_exec_kb = (min_exec_byte + 1023) // 1024
        exec_kb_list = list(range(min_exec_kb, args.core_mem_kb+1, 10)) # 10 -> number of KB incremented each time

        # base_num_thread = (os.cpu_count() or 20)//4
        base_num_thread = (os.cpu_count())//CORE_REDUCE
        base_num_order_per_thread = (len(exec_kb_list) + base_num_thread - 1) // base_num_thread

        base_params = []
        for base_thread in range(base_num_thread):
            exec_kbs = exec_kb_list[base_thread*base_num_order_per_thread:(base_thread+1)*base_num_order_per_thread]
            base_params.append((pickle_filename, args.hbm_bw, layer, exec_kbs))

        
        start = time.perf_counter()
        # with Pool(os.cpu_count()//2 or 20) as base_pool:
        with Pool(os.cpu_count()//8) as base_pool:
            base_results = base_pool.map(search_optimal_exe_load_config_baseline_independent_helper, base_params)
        end = time.perf_counter()
        print(f"\nbaseline schedule time: {end - start} sec")
        log_file.write(f"\nbaseline schedule time: {end - start} sec\n")

        base_results = list(base_results)
        # with open(f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}-{args.hbm_bw}GBps/baseline_results.pickle", 'wb+') as f:
            # pickle.dump(base_results, f)

        baseline_fastest_cycles_min_cold = math.inf
        baseline_fastest_plan_min_cold = (0, [], [], [], [], [], [])
        baseline_fastest_cycles_max_cold = math.inf
        baseline_fastest_plan_max_cold = (0, [], [], [], [], [], [])
        for base_param, base_result_group in zip(base_params, base_results):
            pickle_filename, hbm_bw, layer, exec_kbs = base_param
            for exec_kb, base_result in zip(exec_kbs, base_result_group):
                base_exe_list_temp_min_cold, base_load_list_temp_min_cold, \
                    base_cold_hot_list_temp_min_cold, base_comp_shift_list_temp_min_cold, \
                    base_num_overlap_load_temp_min_cold, base_load_remain_noc_list_temp_min_cold = base_result[0]
                base_exe_list_temp_max_cold, base_load_list_temp_max_cold, \
                    base_cold_hot_list_temp_max_cold, base_comp_shift_list_temp_max_cold, \
                    base_num_overlap_load_temp_max_cold, base_load_remain_noc_list_temp_max_cold = base_result[1]

                if args.report_all_baseline:
                    base_min_cold_exe_cycles = sum([exe[1]-exe[0] for exe in base_exe_list_temp_min_cold])
                    base_min_cold_load_cycles = sum([load[1]-load[0] for load in base_load_list_temp_min_cold])
                    base_min_cold_cycles = -base_load_list_temp_min_cold[0][0]

                    base_min_cold_exe_ms = IPU_Mk2_cycle_to_ms(base_min_cold_exe_cycles)
                    base_min_cold_ms = IPU_Mk2_cycle_to_ms(base_min_cold_cycles)

                    base_max_cold_exe_cycles = sum([exe[1]-exe[0] for exe in base_exe_list_temp_max_cold])
                    base_max_cold_load_cycles = sum([load[1]-load[0] for load in base_load_list_temp_max_cold])
                    base_max_cold_cycles = -base_load_list_temp_max_cold[0][0]

                    base_max_cold_exe_ms = IPU_Mk2_cycle_to_ms(base_max_cold_exe_cycles)
                    base_max_cold_load_ms = IPU_Mk2_cycle_to_ms(base_max_cold_load_cycles)
                    base_max_cold_ms = IPU_Mk2_cycle_to_ms(base_max_cold_cycles)
                    baseline_file.write(f"{exec_kb} KB: {base_min_cold_ms} {base_max_cold_ms}\n")
                    # log_file.write(f"{exec_kb} KB: {base_min_cold_ms} {base_max_cold_ms}\n")

                if len(base_exe_list_temp_min_cold) > 0 and -base_load_list_temp_min_cold[0][0] < baseline_fastest_cycles_min_cold:
                    baseline_fastest_cycles_min_cold = -base_load_list_temp_min_cold[0][0]
                    baseline_fastest_plan_min_cold = (exec_kb, base_exe_list_temp_min_cold, base_load_list_temp_min_cold, 
                                                    base_cold_hot_list_temp_min_cold, base_comp_shift_list_temp_min_cold,
                                                        base_num_overlap_load_temp_min_cold, base_load_remain_noc_list_temp_min_cold)
                if len(base_exe_list_temp_max_cold) > 0 and -base_load_list_temp_max_cold[0][0] < baseline_fastest_cycles_max_cold:
                    baseline_fastest_cycles_max_cold = -base_load_list_temp_max_cold[0][0]
                    baseline_fastest_plan_max_cold = (exec_kb, base_exe_list_temp_max_cold, base_load_list_temp_max_cold, 
                                                    base_cold_hot_list_temp_max_cold, base_comp_shift_list_temp_max_cold,
                                                        base_num_overlap_load_temp_max_cold, base_load_remain_noc_list_temp_max_cold)
        base_min_cold_exec_kb, base_min_cold_exe_list, base_min_cold_load_list, base_min_cold_cold_hot_list, \
            base_min_cold_comp_shift_list, base_min_cold_num_overlap_load, base_min_cold_load_remain_noc_list = baseline_fastest_plan_min_cold
        base_max_cold_exec_kb, base_max_cold_exe_list, base_max_cold_load_list, base_max_cold_cold_hot_list, \
            base_max_cold_comp_shift_list, base_max_cold_num_overlap_load, base_max_cold_load_remain_noc_list = baseline_fastest_plan_max_cold
        
        base_min_cold_exe_cycles = sum([exe[1]-exe[0] for exe in base_min_cold_exe_list])
        base_min_cold_load_cycles = sum([load[1]-load[0] for load in base_min_cold_load_list])
        base_min_cold_cycles = -base_min_cold_load_list[0][0]

        base_min_cold_exe_ms = IPU_Mk2_cycle_to_ms(base_min_cold_exe_cycles)
        base_min_cold_load_ms = IPU_Mk2_cycle_to_ms(base_min_cold_load_cycles)
        base_min_cold_ms = IPU_Mk2_cycle_to_ms(base_min_cold_cycles)

        print(f"base_min_cold_exec_kb: {base_min_cold_exec_kb}")
        print(f"base_min_cold_exe_ms: {base_min_cold_exe_ms}")
        print(f"base_min_cold_load_ms: {base_min_cold_load_ms}")
        print(f"base_min_cold_ms: {base_min_cold_ms}")
        log_file.write(f"base_min_cold_exec_kb: {base_min_cold_exec_kb}\n")
        log_file.write(f"base_min_cold_exe_ms: {base_min_cold_exe_ms}\n")
        log_file.write(f"base_min_cold_load_ms: {base_min_cold_load_ms}\n")
        log_file.write(f"base_min_cold_ms: {base_min_cold_ms}\n")

        if timing_file is not None:
            timing_file.write("Min Cold:\n")
            timing_file.write(str(base_min_cold_exe_list))
            timing_file.write("\n")
            timing_file.write(str(base_min_cold_load_list))
            timing_file.write("\n")
            timing_file.write(str(base_min_cold_comp_shift_list)) # Needed for interconnect utilization
            timing_file.write("\n")
            timing_file.write(str(base_min_cold_load_remain_noc_list)) # Needed for interconnect utilization
            timing_file.write("\n")

        if cold_hot_file is not None:
            cold_hot_file.write("Min Cold:\n")
            cold_hot_file.write(str(base_min_cold_cold_hot_list))
            cold_hot_file.write("\n")

        base_max_cold_exe_cycles = sum([exe[1]-exe[0] for exe in base_max_cold_exe_list])
        base_max_cold_load_cycles = sum([load[1]-load[0] for load in base_max_cold_load_list])
        base_max_cold_cycles = -base_max_cold_load_list[0][0]

        base_max_cold_exe_ms = IPU_Mk2_cycle_to_ms(base_max_cold_exe_cycles)
        base_max_cold_load_ms = IPU_Mk2_cycle_to_ms(base_max_cold_load_cycles)
        base_max_cold_ms = IPU_Mk2_cycle_to_ms(base_max_cold_cycles)

        print(f"\nbase_max_cold_exec_kb: {base_max_cold_exec_kb}")
        print(f"base_max_cold_exe_ms: {base_max_cold_exe_ms}")
        print(f"base_max_cold_load_ms: {base_max_cold_load_ms}")
        print(f"base_max_cold_ms: {base_max_cold_ms}")
        log_file.write(f"\nbase_max_cold_exec_kb: {base_max_cold_exec_kb}\n")
        log_file.write(f"base_max_cold_exe_ms: {base_max_cold_exe_ms}\n")
        log_file.write(f"base_max_cold_load_ms: {base_max_cold_load_ms}\n")
        log_file.write(f"base_max_cold_ms: {base_max_cold_ms}\n")

        if timing_file is not None:
            timing_file.write("Max Cold:\n")
            timing_file.write(str(base_max_cold_exe_list))
            timing_file.write("\n")
            timing_file.write(str(base_max_cold_load_list))
            timing_file.write("\n")
            timing_file.write(str(base_max_cold_comp_shift_list))
            timing_file.write("\n")
            timing_file.write(str(base_max_cold_load_remain_noc_list))
            timing_file.write("\n")

        if cold_hot_file is not None:
            cold_hot_file.write("Max Cold:\n")
            cold_hot_file.write(str(base_max_cold_cold_hot_list))
            cold_hot_file.write("\n")

    # default order
    
    start = time.perf_counter()

    if DNNProgram.MESH < 0.5:
        icbm_ordered_exe_list_1, icbm_ordered_load_list_1, icbm_ordered_cold_hot_list_1, icbm_ordered_comp_shift_list_1, \
            icbm_ordered_num_overlap_load_1, icbm_ordered_load_remain_noc_list_1 = prog.search_optimal_exe_load_config_all(args.hbm_bw, layer, delay_comp=True, load_noc_threshold=0.9)

        icbm_ordered_exe_list_2, icbm_ordered_load_list_2, icbm_ordered_cold_hot_list_2, icbm_ordered_comp_shift_list_2, \
            icbm_ordered_num_overlap_load_2, icbm_ordered_load_remain_noc_list_2 = prog.search_optimal_exe_load_config_all(args.hbm_bw, layer, delay_comp=True, load_noc_threshold=0.7)

        icbm_ordered_exe_list_3, icbm_ordered_load_list_3, icbm_ordered_cold_hot_list_3, icbm_ordered_comp_shift_list_3, \
            icbm_ordered_num_overlap_load_3, icbm_ordered_load_remain_noc_list_3 = prog.search_optimal_exe_load_config_all(args.hbm_bw, layer, delay_comp=True, load_noc_threshold=0.5)

        if -icbm_ordered_load_list_1[0][0] < -icbm_ordered_load_list_2[0][0] and -icbm_ordered_load_list_1[0][0] < -icbm_ordered_load_list_3[0][0]:
            icbm_ordered_exe_list, icbm_ordered_load_list, icbm_ordered_cold_hot_list, icbm_ordered_comp_shift_list, \
                icbm_ordered_num_overlap_load, icbm_ordered_load_remain_noc_list = \
            icbm_ordered_exe_list_1, icbm_ordered_load_list_1, icbm_ordered_cold_hot_list_1, icbm_ordered_comp_shift_list_1, \
                icbm_ordered_num_overlap_load_1, icbm_ordered_load_remain_noc_list_1
        elif -icbm_ordered_load_list_2[0][0] < -icbm_ordered_load_list_1[0][0] and -icbm_ordered_load_list_2[0][0] < -icbm_ordered_load_list_3[0][0]:
            icbm_ordered_exe_list, icbm_ordered_load_list, icbm_ordered_cold_hot_list, icbm_ordered_comp_shift_list, \
                icbm_ordered_num_overlap_load, icbm_ordered_load_remain_noc_list = \
            icbm_ordered_exe_list_2, icbm_ordered_load_list_2, icbm_ordered_cold_hot_list_2, icbm_ordered_comp_shift_list_2, \
                icbm_ordered_num_overlap_load_2, icbm_ordered_load_remain_noc_list_2
        else:
            icbm_ordered_exe_list, icbm_ordered_load_list, icbm_ordered_cold_hot_list, icbm_ordered_comp_shift_list, \
                icbm_ordered_num_overlap_load, icbm_ordered_load_remain_noc_list = \
            icbm_ordered_exe_list_3, icbm_ordered_load_list_3, icbm_ordered_cold_hot_list_3, icbm_ordered_comp_shift_list_3, \
                icbm_ordered_num_overlap_load_3, icbm_ordered_load_remain_noc_list_3
    else:
        icbm_ordered_exe_list, icbm_ordered_load_list, icbm_ordered_cold_hot_list, icbm_ordered_comp_shift_list, \
            icbm_ordered_num_overlap_load, icbm_ordered_load_remain_noc_list = prog.search_optimal_exe_load_config_all(args.hbm_bw, layer, delay_comp=True, load_noc_threshold=0.7)

    # icbm_ordered_exe_list, icbm_ordered_load_list, icbm_ordered_cold_hot_list, icbm_ordered_comp_shift_list, \
    #     icbm_ordered_num_overlap_load, icbm_ordered_load_remain_noc_list = \
    # icbm_ordered_exe_list_2, icbm_ordered_load_list_2, icbm_ordered_cold_hot_list_2, icbm_ordered_comp_shift_list_2, \
    #     icbm_ordered_num_overlap_load_2, icbm_ordered_load_remain_noc_list_2

    end = time.perf_counter()
    print(f"\nicbm_ordered schedule time: {end - start} sec")
    log_file.write(f"\nicbm_ordered schedule time: {end - start} sec\n")

    print("ICBM Ordered hot cold:")
    print(icbm_ordered_cold_hot_list)
    if timing_file is not None:
        timing_file.write("ICBM Ordered:\n")
        timing_file.write(str(icbm_ordered_exe_list))
        timing_file.write("\n")
        timing_file.write(str(icbm_ordered_load_list))
        timing_file.write("\n")
        timing_file.write(str(icbm_ordered_comp_shift_list))
        timing_file.write("\n")
        timing_file.write(str(icbm_ordered_load_remain_noc_list))
        timing_file.write("\n")

    if cold_hot_file is not None:
        cold_hot_file.write("ICBM Ordered:\n")
        cold_hot_file.write(str(icbm_ordered_cold_hot_list))
        cold_hot_file.write("\n")

    icbm_ordered_exe_cycles = sum([exe[1]-exe[0] for exe in icbm_ordered_exe_list])
    icbm_ordered_load_cycles = sum([load[1]-load[0] for load in icbm_ordered_load_list])
    icbm_ordered_cycles = -icbm_ordered_load_list[0][0]

    icbm_ordered_exe_ms = IPU_Mk2_cycle_to_ms(icbm_ordered_exe_cycles)
    icbm_ordered_load_ms = IPU_Mk2_cycle_to_ms(icbm_ordered_load_cycles)
    icbm_ordered_ms = IPU_Mk2_cycle_to_ms(icbm_ordered_cycles)

    # print(f"Delay Comp. False? {IPU_Mk2_cycle_to_ms(-icbm_ordered_load_list_1[0][0])} True? {IPU_Mk2_cycle_to_ms(-icbm_ordered_load_list_2[0][0])}", file=sys.stderr)

    print(f"icbm_ordered_exe_ms: {icbm_ordered_exe_ms}")
    print(f"icbm_ordered_load_ms: {icbm_ordered_load_ms}")
    print(f"icbm_ordered_ms: {icbm_ordered_ms}")
    log_file.write(f"icbm_ordered_exe_ms: {icbm_ordered_exe_ms}\n")
    log_file.write(f"icbm_ordered_load_ms: {icbm_ordered_load_ms}\n")
    log_file.write(f"icbm_ordered_ms: {icbm_ordered_ms}\n")

    # ideal
    
    start = time.perf_counter()
    ideal_exe_list, ideal_load_list, ideal_cold_hot_list, ideal_comp_shift_list, \
        ideal_num_overlap_load, ideal_load_remain_noc_list = prog.ideal_search_optimal_exe_load_config_all(args.hbm_bw, layer)
    end = time.perf_counter()
    print(f"\nideal schedule time: {end - start} sec")
    log_file.write(f"\nideal schedule time: {end - start} sec\n")

    ideal_exe_cycles = sum([exe[1]-exe[0] for exe in ideal_exe_list])
    ideal_load_cycles = sum([load[1]-load[0] for load in ideal_load_list])
    ideal_cycles = -ideal_load_list[0][0]

    ideal_exe_ms = IPU_Mk2_cycle_to_ms(ideal_exe_cycles)
    ideal_load_ms = IPU_Mk2_cycle_to_ms(ideal_load_cycles)
    ideal_ms = IPU_Mk2_cycle_to_ms(ideal_cycles)

    print(f"ideal_exe_ms: {ideal_exe_ms}")
    print(f"ideal_load_ms: {ideal_load_ms}")
    print(f"ideal_ms: {ideal_ms}")
    log_file.write(f"ideal_exe_ms: {ideal_exe_ms}\n")
    log_file.write(f"ideal_load_ms: {ideal_load_ms}\n")
    log_file.write(f"ideal_ms: {ideal_ms}\n")

    if timing_file is not None:
        timing_file.write("Ideal:\n")
        timing_file.write(str(ideal_exe_list))
        timing_file.write("\n")
        timing_file.write(str(ideal_load_list))
        timing_file.write("\n")
        timing_file.write(str(ideal_comp_shift_list))
        timing_file.write("\n")
        timing_file.write(str(ideal_load_remain_noc_list))
        timing_file.write("\n")
    
    # full search
    
    if args.launch_icbm_full:
        del prog
        # num_thread = (os.cpu_count() or 20)
        num_thread = (os.cpu_count() or 20) // CORE_REDUCE * 3
        num_order_per_thread = (len(all_op_orderings) + num_thread - 1) // num_thread

        params = []
        for thread in range(min(num_thread, len(all_op_orderings))):
            orders = all_op_orderings[thread*num_order_per_thread:(thread+1)*num_order_per_thread]
            params.append((pickle_filename, args.hbm_bw, layer, orders))

        start = time.perf_counter()
        # with Pool(os.cpu_count() or 20) as pool:
        with Pool((os.cpu_count() or 20)//CORE_REDUCE) as pool:
            results = pool.map(search_optimal_exe_load_config_order_independent_helper, params)
        end = time.perf_counter()
        print(f"\nicbm schedule time: {end - start} sec")
        log_file.write(f"\nicbm schedule time: {end - start} sec\n")

        results = list(results)
        # os.makedirs(f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}-{args.hbm_bw}GBps", exist_ok=True)
        # with open(f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}-{args.hbm_bw}GBps/results.pickle", 'wb') as f:
            # pickle.dump(results, f)

        fastest_cycles = math.inf
        fastest_plan = ([], [], [], [], [], [], [], None)
        for param, result_group in zip(params, results):
            pickle_filename, hbm_bw, layer, orders = param
            for order, result_pair in zip(orders, result_group):
                for result, delay_comp in zip(result_pair, [False, True]):
                    exe_list, load_list, cold_hot_list, comp_shift_list, num_overlap_load, load_remain_noc_list = result
                    print(exe_list[:5], load_list[:5], delay_comp)
                    if len(exe_list) > 0:
                        total_cycles = -load_list[0][0]
                        if total_cycles < fastest_cycles:
                            fastest_cycles = total_cycles
                            fastest_plan = (order, exe_list, load_list, cold_hot_list, comp_shift_list, num_overlap_load, load_remain_noc_list, delay_comp)
        order, exe_list, load_list, cold_hot_list, comp_shift_list, num_overlap_load, load_remain_noc_list, delay_comp = fastest_plan
        print(load_list[5:], delay_comp)

        def compute_avg_edit_dist(perm):
            dist = [abs(i-j) for i, j in zip(range(len(perm)), perm)]
            avg_dist = sum(dist) / len(dist)
            return avg_dist
        avg_dist = compute_avg_edit_dist(order)
        print(f"avg_edit_dist: {avg_dist}", file=sys.stderr)

        total_exe_cycles = sum([exe[1]-exe[0] for exe in exe_list]) # type: ignore
        total_load_cycles = sum([load[1]-load[0] for load in load_list]) # type: ignore
        total_cycles = -load_list[0][0]

        total_exe_ms = IPU_Mk2_cycle_to_ms(total_exe_cycles)
        total_load_ms = IPU_Mk2_cycle_to_ms(total_load_cycles)
        total_ms = IPU_Mk2_cycle_to_ms(total_cycles)

        print(f"icbm_exe_ms: {total_exe_ms}")
        print(f"icbm_load_ms: {total_load_ms}")
        print(f"icbm_ms: {total_ms}")
        # print(f"\norder: {order}")
        log_file.write(f"icbm_exe_ms: {total_exe_ms}\n")
        log_file.write(f"icbm_load_ms: {total_load_ms}\n")
        log_file.write(f"icbm_ms: {total_ms}\n")
        log_file.write(f"\norder: {order}\n")

        if timing_file is not None:
            timing_file.write("ICBM:\n")
            timing_file.write(str(exe_list))
            timing_file.write("\n")
            timing_file.write(str(load_list))
            timing_file.write("\n")
            timing_file.write(str(comp_shift_list))
            timing_file.write("\n")
            timing_file.write(str(load_remain_noc_list))
            timing_file.write("\n")
            timing_file.write(str(order))
            timing_file.write("\n")

        if cold_hot_file is not None:
            cold_hot_file.write("ICBM:\n")
            cold_hot_file.write(str(cold_hot_list))
            cold_hot_file.write("\n")

    log_file.close()

    if timing_file is not None:
        timing_file.close()

    if baseline_file is not None:
        baseline_file.close()

    if cold_hot_file is not None:
        cold_hot_file.close()
