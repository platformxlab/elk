from typing import List
import itertools
from typing import Any, Dict, List, Optional, Tuple
import pickle
import time

from DNNProgram import DNNProgram, TensorOperator

import TensorExpression as TE

def get_model_from_file(filename: str,
                        num_cores: List[int] = [1472],
                        tot_mem_size_per_core: int = 624 * 1024,
                        name: str = "",
                        output_dir: str = "") -> DNNProgram:
    with open(filename, "r") as f:
        import ujson as json
        # [op_name, dims, variables, ignore_variables, op_type_id]
        ops_arr: List[List] = json.load(f)
        # ops_arr = [op for op in ops_arr if op[4] != TensorExpression.OP_TYPE_GATHER]
    
    ops = [
        TensorOperator(
            name        = f"Op_{op_idx}_{op_list[0]}",
            op_type     = op_list[4],
            dim_lengths = op_list[1],
            variables   = op_list[2],
            num_cores   = num_cores,
            max_byte_per_core   = tot_mem_size_per_core,
            ignore_variables    = [True] + op_list[3],
            output_idx          = op_list[5],
            input_idx_list      = op_list[6],
        ) for op_idx, op_list in enumerate(ops_arr)
    ]

    print(f"Core Utilization Constraint: {TE.CORE_UTIL_THRESHOLD}")
    print(f"Data Padding Constraint: {TE.DATA_PAD_THRESHOLD}")
    print(f"Num Dimensions Correlation: {TE.NUM_DIMS_CORRELATION}")
    
    return DNNProgram(num_cores, tot_mem_size_per_core, ops, name, output_dir)


def search_optimal_exe_load_config_order_independent_helper(param: Tuple[str, float, int, List[List[int]]]) \
        -> List[Tuple[  List[Tuple[float,float]],
                        List[Tuple[float,float]],
                        List[Tuple[int  ,int  ]],
                        List[Tuple[float,float]],
                        List[int], List[float]      ]]:
    
    pickle_filename, hbm_GBps, num_layers, orders = param
    with open(pickle_filename, 'rb') as f:
        prog:DNNProgram = pickle.load(f)
    
    results = []
    for i, order in enumerate(orders):
        start = time.perf_counter()    
        result_delay_load = prog.search_optimal_exe_load_config_order(hbm_GBps, num_layers, order, False)
        result_delay_compute = prog.search_optimal_exe_load_config_order(hbm_GBps, num_layers, order, True)
        end = time.perf_counter()
        # print(f"Step {i} takes {end - start} sec.", flush=True)
        results.append([result_delay_load, result_delay_compute])

    return results

def search_optimal_exe_load_config_baseline_independent_helper(param: Tuple[str, float, int, List[int]]) \
        -> List[List[Tuple[ List[Tuple[float,float]],
                            List[Tuple[float,float]],
                            List[Tuple[int  ,int  ]],
                            List[Tuple[float,float]],
                            List[int], List[float]  ]]]:
    
    pickle_filename, hbm_GBps, num_layers, exe_kb_list = param
    with open(pickle_filename, 'rb') as f:
        prog:DNNProgram = pickle.load(f)
    
    results: List[List[Tuple[List[Tuple[float,float]],
                             List[Tuple[float,float]],
                             List[Tuple[int,int]],
                             List[Tuple[float,float]], 
                             List[int], List[float]]]] = []
    for exe_kb in exe_kb_list:
        result = [prog.baseline_search_optimal_exe_load_config_all(hbm_GBps, num_layers, exe_kb, False),
                    prog.baseline_search_optimal_exe_load_config_all(hbm_GBps, num_layers, exe_kb, True)]
        results.append(result)
    return results
