from copy import deepcopy
from functools import lru_cache
import itertools
from multiprocessing import Pool as DPool
# from OurPool import OurPool as Pool
from concurrent.futures import ProcessPoolExecutor as Pool
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os, sys
import time
import math

import TensorExpression as TE
from TensorExpression import TensorExpression
from OpPartitionSearch import build_spatial_search_tree

from fast_perms import edit_dist_permutations, reduced_edit_dist_permutations

COLD_THRESHOLD = 1.01
# SKIP_OP_THRESHOLD = 1024

SKIP_OP_THRESHOLD = 0

LOAD_NOC_THRESHOLD_NAIVE = 0.7 # ... 0.9 0.7 0.5
LOAD_NOC_THRESHOLD_BASE = 0.7
LOAD_NOC_THRESHOLD_ORDER = 0.7
LOAD_NOC_THRESHOLD_REORDER = 0.99

DELAY_COMP_NAIVE = True
DELAY_COMP_BASE = True
DELAY_COMP_ORDER = False
DELAY_COMP_REORDER = False

GBPS_PER_CORE = 5.5


MESH = 1.0
COMM = 1.0
COMP = 1.0
TRAINING = False


def convert_tuple_to_list(t):
    return [convert_tuple_to_list(x) for x in t] if isinstance(t, tuple) else t

def convert_list_to_tuple(l):
    return tuple(convert_list_to_tuple(x) for x in l) if isinstance(l, list) else l

@lru_cache(maxsize=None)
def create_tensor_expression(dim_lengths: Tuple[int, ...],
                             op_type: int,
                             variables: Tuple[Tuple[Tuple[int, ...], ...], ...],
                             num_cores: Tuple[int, ...] = (),
                             name="",
                             num_byte_per_elem: int = 2,
                             max_byte_per_core: int = 250000,
                             ignore_variables: Optional[Tuple[bool]] = None) -> TensorExpression:
    return TensorExpression(
        op_type,
        convert_tuple_to_list(dim_lengths),
        convert_tuple_to_list(variables),
        convert_tuple_to_list(num_cores),
        name,
        num_byte_per_elem,
        max_byte_per_core,
        convert_tuple_to_list(ignore_variables)
    )

class TensorOperator:
    def __init__(self, name: str, 
                 op_type: int,
                 dim_lengths: Optional[List[int]] = None,
                 variables: Optional[List[List[List[int]]]] = None,
                 num_cores: List[int] = [1472],
                 num_byte_per_elem: int = 2,
                 max_byte_per_core: int = 624 * 1024,
                 ignore_variables: List[bool] = [],
                 output_idx: Optional[int] = None,
                 input_idx_list: Optional[List[int]] = None,
                ):
        self.name: str = name
        if dim_lengths is None or variables is None:
            return
        self.expr: TensorExpression = create_tensor_expression(
            convert_list_to_tuple(dim_lengths),
            op_type,
            convert_list_to_tuple(variables),
            convert_list_to_tuple(num_cores),
            "",
            num_byte_per_elem,
            max_byte_per_core,
            convert_list_to_tuple(ignore_variables)
        )
    
        self.output_idx: Optional[int] = output_idx
        self.input_idx_list: Optional[List[int]] = input_idx_list

        self.dim_lengths: List[int] = dim_lengths
        self.op_type = op_type
        self.variables: List[List[List[int]]] = variables
        self.num_cores: List[int] = num_cores
        self.num_byte_per_elem: int = num_byte_per_elem
        self.max_byte_per_core: int = max_byte_per_core
        self.ignore_variables: List[bool] = ignore_variables
        self.ignore_variables_unconfirmed: Optional[List[bool]] = ignore_variables
        
        self.min_hot_size_bytes_per_core: Optional[int] = None
        '''min hot size (sum of all variables) in bytes'''
        self.min_cold_size_bytes_per_core: int = 0
        '''min cold size (sum of all variables) in bytes'''
        


class DNNProgram:

    def __init__(self, num_cores: Optional[List[int]] = None, tot_mem_size_per_core: int = 624 * 1024, ops: Optional[List[TensorOperator]] = None, name: str = "", output_dir: str = ""):
        if ops is None:
            ops = []
        if num_cores is None:
            num_cores = [1472]

        self.name: str = name

        self.op_groups: List[List[TensorOperator]] = [ops]
        for op in self.ops:
            op.expr.output_dir = os.path.join(output_dir, name)

        self.num_cores: List[int] = num_cores
        self.tot_num_cores: int = int(np.prod(num_cores))
        self.spatial_search_tree: Dict = {}
        self.tot_mem_size_per_core: int = tot_mem_size_per_core
        self.op_execution_plan: List[Dict[str, List]] = []
        '''[op group 0 plan {op 0 name: exe plan ((cold plan, (mem size, exe time)), (hot plan, (mem size, exe time)), exe time), op 1 name: exe plan, ...}, op group 1 plan, ...]'''

        self.spatial_search_tree_time: Dict[Any, float] = {}
        '''{(depth, tot_dim_size, dim_size_TH): search time (s)}'''
        self.intra_op_compile_time: Dict[str, float] = {}
        '''{op name: compile time (s)}'''
        self.cold_hot_table_compile_time: Dict[str, float] = {}
        '''{op name: compile time (s)}'''
        self.inter_op_compile_time: Dict[int, float] = {}
        '''{op group (index): compile time (s)}'''
        self.all_order_lists: List[List[int]] = []

        self.output_dir: str = os.path.join(output_dir, name)
        os.makedirs(self.output_dir, exist_ok=True)



    @property
    def ops(self) -> List[TensorOperator]:
        return list(itertools.chain.from_iterable(self.op_groups))
    
    # def get_TH(self, op: TensorOperator, tier: int=20) -> float:
    #     dim_size_TH = TE.CORE_UTIL_THRESHOLD * op.expr.get_util_threshold()
    #     dim_size_TH = np.floor(dim_size_TH*tier) / tier
    #     return dim_size_TH

    def get_op_spatial_search_tree(self, op: TensorOperator):
        return self.get_op_spatial_search_tree_TExpr(op.expr)

    def get_op_spatial_search_tree_TExpr(self, expr: TensorExpression):
        depth = len(expr.dim_lengths) - 1
        tot_dim_size = int(np.prod(self.num_cores))
        dim_size_TH = expr.get_dim_size_threshold()
        return self.spatial_search_tree[(depth, tot_dim_size, dim_size_TH)]

    # def build_spatial_search_tree_helper(self, param):
    #     return build_spatial_search_tree(*param)

    # def generate_all_spatial_search_trees(self, num_threads: int = 1):
    #     # depth = len(self.dim_lengths) - 1,
    #     # tot_dim_size = int(np.prod(self.num_cores)),
    #     # dim_size_TH = CORE_UTIL_THRESHOLD
    #     params = set()
    #     for op in self.ops:
    #         depth = len(op.expr.dim_lengths) - 1
    #         tot_dim_size = int(np.prod(op.expr.num_cores))
    #         dim_size_TH = op.expr.get_dim_size_threshold()
    #         num_core = np.prod(op.expr.num_cores)
    #         if (depth, tot_dim_size, dim_size_TH) not in self.spatial_search_tree:
    #             params.add((depth, tot_dim_size, None, None, dim_size_TH, num_core))
        
    #     if len(params) == 0:
    #         return
        
    #     print("Building spatial search trees... (", len(params), "trees)")
    #     params = list(params)
    #     with DPool(min(len(params), num_threads)) as p:
    #         spatial_search_tree_list = p.map(self.build_spatial_search_tree_helper, params)

    #     for param, (compile_time, tree) in zip(params, spatial_search_tree_list):
    #         self.spatial_search_tree[(param[0], param[1], param[4])] = tree
    #         self.spatial_search_tree_time[(param[0], param[1], param[4])] = compile_time
        
    #     with open(f"{self.output_dir}/spatial_search_tree_time.json", "w") as f:
    #         import ujson as json
    #         json.dump(self.spatial_search_tree_time, f, indent=4)
        
    #     print("Done: Building spatial search trees.", flush=True)

    def get_unique_expr_to_opnames_dict(self, intra:bool) -> Dict[TensorExpression, List[str]]:
        expr_to_opnames_dict = {}
        opnames_to_first_name_dict = {}
        for op in self.ops:
            if op.expr not in expr_to_opnames_dict:
                expr_to_opnames_dict[op.expr] = []
                op.expr.name = op.name
            expr_to_opnames_dict[op.expr].append(op.name)
            opnames_to_first_name_dict[op.name] = op.expr.name

        if intra:
            with open(f"{self.output_dir}/all_configs_dict.json", "w") as f:
                import ujson as json
                json.dump(opnames_to_first_name_dict, f, indent=4)
        else:
            with open(f"{self.output_dir}/cold_hot_table_dict.json", "w") as f:
                import ujson as json
                json.dump(opnames_to_first_name_dict, f, indent=4)
        return expr_to_opnames_dict

    def run_intra_op_optimization_for_op(self, params: Tuple[TensorExpression, List[str], int]):
        """Returns (expr, opnames, compile_time)"""
        expr, opnames, num_threads = params
        print("### Optimizing op: ", opnames, "...", flush=True)
        start = time.perf_counter()
        # expr.search_optimal_config(num_threads, self.get_op_spatial_search_tree_TExpr(expr), log_filename=opnames[0])
        expr.search_optimal_config(num_threads, None, log_filename=opnames[0])
        end = time.perf_counter()
        # self.intra_op_compile_time[op.name] = end - start
        # print(flush=True)
        return expr, opnames, end - start

    def update_ops_and_compile_time(self, results, compile_time_dict):
        results = list(results)
        # get compile time
        for expr, opnames, compile_time in results:
            for opname in opnames:
                compile_time_dict[opname] = compile_time / len(opnames)
        # get ops
        for expr, opnames, _ in results:
            # search through all groups
            for op_group in self.op_groups:
                # update op
                for i in range(len(op_group)):
                    for opname in opnames:
                        if opname == op_group[i].name:
                            op_group[i].expr = expr

    def run_intra_op_optimization(self, num_threads: int = 1):
        # self.generate_all_spatial_search_trees(num_threads)
        # print()

        print("# Running intra-op optimization...", flush=True)
        # for op in self.ops:
        #     print("### Optimizing op: ", op.name, "...")
        #     start = time.perf_counter()
        #     op.expr.search_optimal_config(num_threads, self.get_op_spatial_search_tree(op), log_filename=op.name)
        #     end = time.perf_counter()
        #     self.intra_op_compile_time[op.name] = end - start
        #     print()
        params = [(expr, names, num_threads) for expr, names in self.get_unique_expr_to_opnames_dict(intra=True).items()]
        with Pool(min(len(params), num_threads)) as p:
            results = p.map(self.run_intra_op_optimization_for_op, params)
        self.update_ops_and_compile_time(results, self.intra_op_compile_time)
        print("# Done: intra-op optimization.", flush=True)

        with open(f"{self.output_dir}/intra_op_compile_time.json", "w") as f:
            import ujson as json
            json.dump(self.intra_op_compile_time, f, indent=4)

    def generate_cold_hot_table_for_op(self, params: Tuple[TensorExpression, List[str], int, float]):
        """Returns (expr, opnames, compile_time)"""
        expr, opnames, num_threads, cold_hot_threshold = params
        print("### Generating cold-hot table for op: ", opnames, "...", flush=True)
        start = time.perf_counter()
        # expr.search_optimal_config_cold(num_threads, self.get_op_spatial_search_tree_TExpr(expr), log_filename=opnames[0])
        expr.search_optimal_config_cold(num_threads, None, log_filename=opnames[0])
        expr.generate_cold_hot_table(num_threads, cold_hot_threshold, log_filename=opnames[0])
        end = time.perf_counter()
        print("### Done: Generating cold-hot table for op: ", opnames, "...", flush=True)
        # self.cold_hot_table_compile_time[op.name] = end - start
        return expr, opnames, end - start

    def generate_all_cold_hot_table(self, num_threads: int, cold_hot_threshold: float = 1):
        # for op in self.ops:
        #     print("### Generating cold-hot table for op: ", op.name, "...")
        #     start = time.perf_counter()
        #     op.expr.search_optimal_config_cold(num_threads, self.get_op_spatial_search_tree(op), log_filename=op.name)
        #     op.expr.generate_cold_hot_table(num_threads, cold_hot_threshold, log_filename=op.name)
        #     end = time.perf_counter()
        #     self.cold_hot_table_compile_time[op.name] = end - start
        #     print()
        print("# Generating cold-hot table for all ops...", flush=True)
        params = [(expr, names, num_threads, cold_hot_threshold) for expr, names in self.get_unique_expr_to_opnames_dict(intra=False).items()]
        with Pool(min(len(params), num_threads)) as p:
            results = p.map(self.generate_cold_hot_table_for_op, params)
        self.update_ops_and_compile_time(results, self.cold_hot_table_compile_time)
        print("# Done: cold-hot table generation for all ops.", flush=True)

        with open(f"{self.output_dir}/cold_hot_table_compile_time.json", "w") as f:
            import ujson as json
            json.dump(self.cold_hot_table_compile_time, f, indent=4)

    def get_best_op_to_advance(self, cur_plan, all_op_next_cold_sizes: List[int], ops: List[TensorOperator]) -> Tuple[int, float]:
        best_op_idx = -1
        best_ratio = float("inf")
        for i, new_cold_size in enumerate(all_op_next_cold_sizes):
            if new_cold_size == -1:
                continue
            
            # compute exe time increase of this op
            old_cold_size = cur_plan[0][i][0]
            old_ref_plan = next(reversed(ops[i].expr.cold_hot_table[old_cold_size].values()))
            old_exe_time = old_ref_plan[2]

            new_ref_plan = next(reversed(ops[i].expr.cold_hot_table[new_cold_size].values()))
            new_exe_time = new_ref_plan[2]

            exe_time_increase = new_exe_time - old_exe_time
            if exe_time_increase <= 0:
                # this op does not decrease its exe time even with smaller cold size, so choose this one
                return i, 0
            # assert exe_time_increase > 0, f"exe time increase should be positive: {exe_time_increase}, old_exe_time: {old_exe_time}, new_exe_time: {new_exe_time}"
            hot_size_increase = old_cold_size - new_cold_size
            assert hot_size_increase > 0, f"hot size increase should be positive: {hot_size_increase}, old_cold_size: {old_cold_size}, new_cold_size: {new_cold_size}"

            # compute the ratio and update the best op
            ratio = exe_time_increase / hot_size_increase
            if ratio < best_ratio:
                best_ratio = ratio
                best_op_idx = i

        assert best_op_idx >= 0 and best_ratio < float("inf"), f"best_op_idx: {best_op_idx}, best_ratio: {best_ratio}"

        return best_op_idx, best_ratio


    def search_optimal_global_config_icbm(self, op_groups: List[List[TensorOperator]], cold_group: List[float] = [], ideal: bool = False):
        # [op_group 0 plan, ...]
        # assert len(op_groups) > 0, "op_groups should not be empty"
        op_group_plans: List[Tuple[List[Tuple[int, int]],]] = []
        bw_per_core_byte_per_cycle = GBPS_PER_CORE * 1024 * 1024 * 1024 / 1.325e9
        def mesh_dict(mesh):
            minor = 4*(mesh-1)
            return max(0, 1-minor)

        for ops in op_groups:
            
            load_cold_size = 0
            for op_idx, op in enumerate(ops[1:]):
                cold_size = op.min_cold_size_bytes_per_core
                if cold_group != []:
                    # if cold_size > 0:
                    #     print(cold_group[op_idx]/cold_size, end=" ")
                    cold_size = (cold_size * cold_group[op_idx])**0.5
                load_cold_size += cold_size
                
            max_hot_size = self.tot_mem_size_per_core - (COLD_THRESHOLD * load_cold_size)
            
            min_cold_size = next(iter(ops[0].expr.cold_hot_table))
            min_hot_size = next(iter(ops[0].expr.cold_hot_table[min_cold_size]))

            if min_hot_size > max_hot_size:
                # assert len(ops)>1, f"min_hot_size: {min_hot_size}, max_hot_size: {max_hot_size}"
                return op_group_plans

            best_exe_time = math.inf
            best_cold_hot = (0,0)
            for cold in ops[0].expr.cold_hot_table:
                if cold > max_hot_size:
                    break
                for hot in ops[0].expr.cold_hot_table[cold]:
                    if hot > max_hot_size:
                        break
                    
                    plan = ops[0].expr.cold_hot_table[cold][hot]
                    if TRAINING:
                        if ideal:
                            load_time_noc = 0
                        else:
                            load_time_noc = cold / bw_per_core_byte_per_cycle / 2
                        exe_time = plan[2] + load_time_noc
                        if exe_time < best_exe_time:
                            best_exe_time = exe_time
                            best_cold_hot = (cold, hot)
                    else:
                        shift_time = plan[1][1][3]
                        comp_time = plan[1][1][2]
                        new_shift_time = shift_time/COMM
                        new_comp_time = comp_time/COMP
                        diff = (shift_time-new_shift_time) + (comp_time-new_comp_time)
                        shift_time = new_shift_time
                        comp_time = new_comp_time
                        if ideal:
                            load_time_noc = 0
                        elif MESH:
                            load_time_noc = shift_time*mesh_dict(MESH)
                        else:
                            load_time_noc = cold / bw_per_core_byte_per_cycle / 2
                        exe_time = plan[2] + load_time_noc - diff
                        if exe_time < best_exe_time:
                            best_exe_time = exe_time
                            best_cold_hot = (cold, hot)
                            
            if best_exe_time == math.inf:
                # assert len(ops)>1, f"min_hot_size: {min_hot_size}, max_hot_size: {max_hot_size}"
                return op_group_plans

            op_group_plans.append(([best_cold_hot]+[(0,0)]*(len(ops)-1),))

        # assert len(op_groups[0])>1 or len(op_group_plans), f"min_hot_size: {min_hot_size}, max_hot_size: {max_hot_size}"
        return op_group_plans


    def search_optimal_global_config_heuristic(self, op_groups: List[List[TensorOperator]]):
        # [op_group 0 plan, ...]
        op_group_plans: List[Tuple[List[Tuple[int, int]], int, int, float, float, float, int]] = []

        for op_group_idx, ops in enumerate(op_groups):
            # print("## Searching for best global cold-hot config for op group: ", op_group_idx)
            # print("## Total Search Space Size (product of cold sizes for all ops):", np.prod(np.array([len(op.expr.cold_hot_table) for op in ops], dtype=np.uint64)))

            # [(inter-op plan 0 [op0 plan (cold, hot), op1 plan (cold, hot), ...], tot cold size, tot size (max(hot-cold)+sum(cold)), e2e exe time, comp time, shift time), ...]
            # ops_cold_hot_plans[inter-op plan index][{plan, cold size, tot size, e2e exe time, comp time, shift time, min cold size}][op index][{cold size, hot size}]
            ops_cold_hot_plans: List[Tuple[List[Tuple[int, int]], int, int, float, float, float, int]] = []

            ##### Strategy: Iteratively reduce cold size for each op to find the best inter-op plan.
            #####   In each iteration, we pick the op that has the most benefit to progress the inter-op plan.
            #####   The insight is that we increase the chosen op's warm-up time by reducing its cold size,
            #####   but this will reduce the exe time of all other ops by increasing their hot sizes.
            #####   We choose the op that has the best (smallest) (warm-up time increase / hot size increase) ratio.
            #####   
            #####   Each iteration explores an inter-op execution plan.
            #####   We stop the iterative process until all ops have the smallest cold size.
            #####   Then, we pick the best plan among all iterations.
            
            # initial cold-hot plan that has the largest cold size for each op
            op_plans = [
                (
                    next(reversed(
                        op.expr.cold_hot_table
                    )),
                    next(reversed(
                        op.expr.cold_hot_table[next(reversed(op.expr.cold_hot_table))]
                    ))
                ) for op in ops
            ]
            cold_size = sum([cold for cold, hot in op_plans])
            tot_size = max([hot - cold for cold, hot in op_plans]) + cold_size
            exe_time = sum([op.expr.cold_hot_table[cold][hot][2] for op, (cold, hot) in zip(ops, op_plans)])
            comp_cycles = sum([op.expr.cold_hot_table[cold][hot][1][1][2] for op, (cold, hot) in zip(ops, op_plans)])
            shift_cycles = sum([op.expr.cold_hot_table[cold][hot][1][1][3] for op, (cold, hot) in zip(ops, op_plans)])
            min_cold_size = sum([op.min_cold_size_bytes_per_core for op in ops if op.min_cold_size_bytes_per_core is not None])
            # (op_plans, cold_size, tot_size, exe_time, comp_cycles, shift_cycles, min_cold_size)
            cur_plan: Tuple[List[Tuple[int, int]], int, int, float, float, float, int] = (op_plans, cold_size, tot_size, exe_time, comp_cycles, shift_cycles, min_cold_size)

            # (cold size, exe time, is valid)
            this_group_search_iter_trace: List[Tuple[int, float, bool]] = []

            # otherwise, we need to iteratively reduce cold size for each op to find the best plan
            if cur_plan[2] <= self.tot_mem_size_per_core:
                ops_cold_hot_plans.append(deepcopy(cur_plan))
            # init the cold size candidates for each op (the largest size is excluded because it is already in cur_plan)
            # cold_size_candidates[op index] is a list of cold sizes for the op in ascending order
            cold_size_candidates = [
                list(op.expr.cold_hot_table.keys())[:-1] for op in ops
            ]

            # iterative search!
            iter_num = 0
            # ffff_num = 0
            while True:
                # collect exe_time vs cold mem size info for each iteration
                this_group_search_iter_trace.append((cur_plan[1], cur_plan[3], cur_plan[2] <= self.tot_mem_size_per_core))

                # print([(cold, hot) for cold, hot in cur_plan[0]])

                ### 0. check if all candidates have been searched
                if all([len(candidates) == 0 for candidates in cold_size_candidates]):
                    # if ffff_num == iter_num:
                    #     print(f"last plan: {cur_plan}")
                    break # no candidates left, we are done
                
                ### 1. mutate the execution plan by reducing cold size for the chosen op
                
                ## 1.1. find the op that has the best (smallest) (warm-up time increase / hot size increase) ratio
                
                # get the possible next cold size for each op
                all_op_next_cold_sizes = [
                    colds[-1] if len(colds) > 0 else -1
                        for i, colds in enumerate(cold_size_candidates)
                ]
                
                # find the best op to advance the plan
                best_op_idx, best_ratio = self.get_best_op_to_advance(cur_plan, all_op_next_cold_sizes, ops)

                ## 1.2. update the execution plan
                
                # update new cold size for the chosen op
                cur_plan[0][best_op_idx] = (all_op_next_cold_sizes[best_op_idx], -1)
                new_max_hot_size = max(0, self.tot_mem_size_per_core - sum([cold for cold, hot in cur_plan[0]]))

                # update hot sizes for all ops
                for i, op in enumerate(ops):
                    new_cold_size = cur_plan[0][i][0] # if i == best_op_idx else all_op_next_cold_sizes[best_op_idx]
                    new_hot_size = op.expr.get_best_hot_size_for_cold(new_cold_size, new_max_hot_size)
                    assert new_hot_size is not None, f"new_hot_size should not be None, new_cold_size: {new_cold_size}, new_max_hot_size: {new_max_hot_size}"
                    cur_plan[0][i] = (new_cold_size, new_hot_size)
                
                # print the selected op
                # print(f"iter {iter_num}: selected op {best_op_idx} with ratio {best_ratio}; new plan for this op: {cur_plan[0][best_op_idx]}")

                # update new tot cold size, tot size, and exe time
                new_tot_cold_size = sum([cold for cold, hot in cur_plan[0]])
                cur_plan = (
                    cur_plan[0],
                    new_tot_cold_size,
                    new_max_hot_size + new_tot_cold_size,
                    sum([op.expr.cold_hot_table[cold][hot][2] for op, (cold, hot) in zip(ops, cur_plan[0])]),
                    sum([op.expr.cold_hot_table[cold][hot][1][1][2] for op, (cold, hot) in zip(ops, cur_plan[0])]),
                    sum([op.expr.cold_hot_table[cold][hot][1][1][3] for op, (cold, hot) in zip(ops, cur_plan[0])]),
                    sum([op.min_cold_size_bytes_per_core for op in ops if op.min_cold_size_bytes_per_core is not None])
                )

                # print(f"op group {op_group_idx}, iter {iter_num}, best op: {best_op_idx}, best ratio: {best_ratio}")
                # print(f"\tcur_plan: {cur_plan}")
                # print(f"\tvalid: {cur_plan[2] <= self.tot_mem_size_per_core}, total mem size: {self.tot_mem_size_per_core}")

                ### 3. remove the cold size candidate of the chosen op
                cold_size_candidates[best_op_idx].pop()

                ### 4. put the new plan into list if this is a valid plan (tot size <= available memory)
                if cur_plan[2] <= self.tot_mem_size_per_core:
                    ops_cold_hot_plans.append(deepcopy(cur_plan))
                
                iter_num += 1
            
            # print(f"op group {op_group_idx} done: # search iterations: {iter_num}, # plans found: {len(ops_cold_hot_plans)}")
            # if len(ops_cold_hot_plans) == 0:
            #     print(f"op group {op_group_idx} done: no valid plan found")
            # print(f"# overflowed cold plan combinations: {ffff_num}")

            # find the best plan among all iterations
            if len(ops_cold_hot_plans) > 0:
                best_plan = min(ops_cold_hot_plans, key=lambda plan: plan[3])
                op_group_plans.append(best_plan)
            else:
                return op_group_plans

        return op_group_plans
    

    def update_num_ipus(self, num_ipus: int, new_num_ipus: int):
        self.name = self.name.replace(f"{num_ipus}ipus", f"{new_num_ipus}ipus")
        # self.name = self.name.replace(output_dir, new_output_dir)
        self.output_dir = self.output_dir.replace(f"{num_ipus}ipus", f"{new_num_ipus}ipus")
        # self.output_dir = self.output_dir.replace(output_dir, new_output_dir)
        for op in self.ops:
            op.expr.name = op.expr.name.replace(f"{num_ipus}ipus", f"{new_num_ipus}ipus")
            # op.expr.name = op.expr.name.replace(output_dir, new_output_dir)
            op.expr.output_dir = op.expr.output_dir.replace(f"{num_ipus}ipus", f"{new_num_ipus}ipus")
            # op.expr.output_dir = op.expr.output_dir.replace(output_dir, new_output_dir)


    def baseline_cold_hot(self, op: TensorOperator, exe_byte_per_core: int, use_largest_cold: bool):
        if use_largest_cold:
            best_cold = -1
            for cold_size in reversed(op.expr.cold_hot_table):
                first_hot = next(iter(op.expr.cold_hot_table[cold_size]))
                if first_hot <= exe_byte_per_core:
                    best_cold = cold_size
                    break
            assert best_cold != -1, "exec size too small."
            best_hot = -1
            for hot_size in op.expr.cold_hot_table[best_cold]:
                if hot_size <= exe_byte_per_core:
                    best_hot = hot_size
                else:
                    break
            return best_cold, best_hot
        else:
            min_cold = next(iter(op.expr.cold_hot_table))
            best_hot = -1
            for hot_size in op.expr.cold_hot_table[min_cold]:
                if hot_size <= exe_byte_per_core:
                    best_hot = hot_size
                else:
                    break
            assert best_hot != -1, "exec size too small."
            return min_cold, best_hot






    #################################### ICBM Functions ####################################






    def search_optimal_exe_load_config_all(self, hbm_GBps: float, num_layers: int,
                                           delay_comp: bool = DELAY_COMP_ORDER,
                                           load_noc_threshold: float = LOAD_NOC_THRESHOLD_ORDER) \
        -> Tuple[ List[Tuple[float,float]], 
                  List[Tuple[float,float]], 
                  List[Tuple[int  ,int  ]],
                  List[Tuple[float,float]],
                  List[int], List[float] ]:

        # self.init_min_cold_size_bytes_per_core()
        plan_dict = {}  # {op_group_keys: op_group_plan} used to cache the search result for each op group
        max_load = len(self.ops) // num_layers  # never pre-load more than a layer of ops
        noc_hbm_bw_ratio = (int(GBPS_PER_CORE)*self.tot_num_cores) / hbm_GBps

        # best exe plan for the last op
        op_group_plans = self.search_optimal_global_config_icbm([self.ops[-1:]])
        assert len(op_group_plans) == 1, "length of op_group_plans should be 1"

        # exe time of the last op
        cold, hot = op_group_plans[0][0][0]
        next_exe_time_cycle:float = self.ops[-1].expr.cold_hot_table[cold][hot][2]
        comp_cycles = self.ops[-1].expr.cold_hot_table[cold][hot][1][1][2]
        shift_cycles = self.ops[-1].expr.cold_hot_table[cold][hot][1][1][3]
        new_shift_cycles = shift_cycles/COMM
        next_exe_time_cycle -= (shift_cycles-new_shift_cycles)
        shift_cycles = new_shift_cycles
        new_comp_cycles = comp_cycles/COMP
        next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
        comp_cycles = new_comp_cycles

        # hbm latency of last op
        next_load_size_byte = self.ops[-1].min_cold_size_bytes_per_core * self.tot_num_cores
        next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
        next_load_time_cycle = next_load_time_ms * 1.325e6
        
        # calculate the noc time required to deliver data to cores, when the last op is loading
        if self.ops[-1].min_cold_size_bytes_per_core > SKIP_OP_THRESHOLD:
            if MESH:
                next_load_noc_cycle = next_load_time_cycle/MESH
            else:
                broadcast_ratio = cold / self.ops[-1].min_cold_size_bytes_per_core
                next_load_noc_cycle = next_load_time_cycle * broadcast_ratio / noc_hbm_bw_ratio
                next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle/load_noc_threshold)
        else:
            next_load_noc_cycle = 0
        # real load time is the max of hbm load time and required noc time

        # initialize the lists using the last op's only possible plan
        exe_list = [(-next_exe_time_cycle, 0.)]
        load_list = [(-next_exe_time_cycle-next_load_time_cycle, -next_exe_time_cycle)]
        num_overlap_load = [0]
        cold_hot_list = [(cold, hot)]
        comp_shift_list = [(comp_cycles, shift_cycles)]

        # the remaining noc time that can be used for shifting
        load_remain_noc_list = [next_load_time_cycle-next_load_noc_cycle]

        while len(exe_list) < len(self.ops):
            next_op_idx = len(self.ops) - len(exe_list) - 1
            if MESH>=1 and MESH<=1.15:
                rate = MESH**13
                delay_comp = next_op_idx%int(32//rate)==0

            next_exe_deadlines = []
            op_group_list:List[List[TensorOperator]] = []
            cold_group:List[float] = []
            cur_exe_start, cur_exe_end = exe_list[-1]
            # find all possible deadlines for the exe of next op, and corresponding ops to overlap with
            for load_i in range(min(len(load_list), max_load)):
                rev_load_i = len(load_list) - 1 - load_i
                op_i = next_op_idx + load_i + 1
                cur_load_start, cur_load_end = load_list[rev_load_i]
                cur_load_cold, cur_load_hot = cold_hot_list[rev_load_i]
                cold_group.append(cur_load_cold)
                
                if cur_load_start >= cur_exe_start:
                    next_exe_deadlines.append(cur_exe_start)
                    op_group_list.append(self.ops[next_op_idx:op_i])
                    break
                elif self.ops[op_i].min_cold_size_bytes_per_core < SKIP_OP_THRESHOLD:
                    continue
                else:
                    next_exe_deadlines.append(cur_load_start)
                    op_group_list.append(self.ops[next_op_idx:op_i])

            # if the last op is not loading data, it is small and elementwise, so we allow it to start late and overlap more
            if self.ops[next_op_idx].min_cold_size_bytes_per_core == 0:
                next_exe_deadlines_back = next_exe_deadlines
                op_group_list_back = op_group_list
                next_exe_deadlines = next_exe_deadlines[-1:]
                op_group_list = op_group_list[-1:]

            # use inter-op optimization to find the best plan for each op overlap plan
            # each op overlap plan is a op group list [exe_op, load_op1, load_op2, ...]
            op_group_plans = []
            
            for op_group in op_group_list:
                op_group_keys = tuple([op.expr.log_filename_physical for op in op_group])
                # use cached result if possible
                if op_group_keys in plan_dict:
                    op_group_plan = plan_dict[op_group_keys]
                else:
                    op_group_plan = self.search_optimal_global_config_icbm([op_group], cold_group)
                    plan_dict[op_group_keys] = op_group_plan
                if op_group_plan == []:
                    break
                else:
                    op_group_plans += op_group_plan
            
            while len(op_group_plans) == 0 and len(op_group_list[0])>1:
                next_exe_deadlines = next_exe_deadlines_back[-len(next_exe_deadlines)-1:]
                op_group_list = op_group_list_back[-len(op_group_list)-1:]
                for op_group in op_group_list:
                    op_group_keys = tuple([op.expr.log_filename_physical for op in op_group])
                    # use cached result if possible
                    if op_group_keys in plan_dict:
                        op_group_plan = plan_dict[op_group_keys]
                    else:
                        op_group_plan = self.search_optimal_global_config_icbm([op_group], cold_group)
                        plan_dict[op_group_keys] = op_group_plan
                    if op_group_plan == []:
                        break
                    else:
                        op_group_plans += op_group_plan

            # remove op groups that have no valid plan
            assert len(op_group_plans)>0, f"op_group_plans should not be empty, op_group_list: {op_group_list}"
            next_exe_deadlines = next_exe_deadlines[:len(op_group_plans)]

            # find the real exe time of each plan, with the noc contention considered
            next_exe_start_times = []
            cold_hot_plans = []
            for op_group, deadline in zip(op_group_plans, next_exe_deadlines):
                cold, hot = op_group[0][0]
                cold_hot_plans.append((cold, hot))

                # get the default start time of exe
                next_exe_time_cycle = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][2]

                shift_cycles_to_handle = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][3]
                new_shift_cycles = shift_cycles_to_handle/COMM
                next_exe_time_cycle -= (shift_cycles_to_handle-new_shift_cycles)
                shift_cycles_to_handle = new_shift_cycles

                comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][2]
                new_comp_cycles = comp_cycles/COMP
                next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
                comp_cycles = new_comp_cycles

                next_exe_start_time = deadline - next_exe_time_cycle

                if delay_comp:
                    # get the noc time required for exe next op
                    # shift_cycles_to_handle = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][3]
                    # get number of preloaded ops
                    num_load_ops = len(op_group[0])-1
                    next_exe_start_time_noc = 1
                    for load_i in range(num_load_ops):
                        # start from the lastest overlapped op
                        rev_load_i_ = len(load_list) - num_load_ops + load_i
                        # remaining noc time reserved for shifting
                        noc_cycle = load_remain_noc_list[rev_load_i_]
                        # end time of the remaining noc time
                        load_start, load_end = load_list[rev_load_i_]
                        # update the two if the load is partially overlapped with the exe
                        if load_end > deadline:
                            adjusted_noc_cycle = noc_cycle * (deadline - load_start) / (load_end - load_start)
                            assert adjusted_noc_cycle >= 0, "adjusted_noc_cycle should be non-negative"
                            adjusted_noc_end = deadline
                        else:
                            adjusted_noc_cycle = noc_cycle
                            adjusted_noc_end = load_end
                        # fit in shift to available noc time
                        shift_cycles_to_handle -= adjusted_noc_cycle
                        # get the latest start time of the exe of next op, considering the noc contention
                        if shift_cycles_to_handle <= 0:
                            shift_cycles_to_handle += adjusted_noc_cycle
                            adjusted_load_duration = adjusted_noc_end-load_start
                            adjusted_shift_cycles = adjusted_load_duration * (shift_cycles_to_handle/adjusted_noc_cycle)
                            next_exe_start_time_noc = adjusted_noc_end-adjusted_shift_cycles
                            break
                    # if shift cannot fit in existing loads, start before the beginning of the last load
                    if next_exe_start_time_noc == 1:
                        next_exe_start_time_noc = load_list[-1][0]-shift_cycles_to_handle
                    next_exe_start_times.append(min(next_exe_start_time, next_exe_start_time_noc))
                else:
                    next_exe_start_times.append(next_exe_start_time)

            # pick latest start time for the exe of next op
            best_num_load = np.argmax(next_exe_start_times)
            cold, hot = cold_hot_plans[best_num_load]

            if not delay_comp:
                # post-process the load time
                comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][2]
                comp_cycles = comp_cycles/COMP
                num_load_ops = len(op_group_plans[best_num_load][0])-1
                last_load_i = -1
                last_load_remain = -1
                for load_i in range(num_load_ops):
                    rev_load_i_ = len(load_list) - num_load_ops + load_i
                    load_start, load_end = load_list[rev_load_i_]
                    load_duration = load_end - load_start
                    noc_cycle = load_duration - load_remain_noc_list[rev_load_i_]
                    if load_end > next_exe_deadlines[best_num_load]:
                        adjusted_noc_end = next_exe_deadlines[best_num_load]
                        adjusted_noc_duration = adjusted_noc_end - load_start
                        if noc_cycle == 0:
                            adjusted_noc_cycle = 0
                        else:
                            adjusted_noc_cycle = noc_cycle * adjusted_noc_duration / load_duration
                    else:
                        adjusted_noc_cycle = noc_cycle
                        adjusted_noc_end = load_end
                        adjusted_noc_duration = load_duration
                    comp_cycles -= adjusted_noc_cycle
                    if comp_cycles <= 0:
                        last_load_i = rev_load_i_
                        last_load_remain = (-comp_cycles) * (load_duration/noc_cycle)
                        break
                    if load_start < next_exe_start_times[best_num_load]:
                        break
                if last_load_i != -1:
                    last_load_start, last_load_end = load_list[last_load_i]
                    new_last_load_start = min(last_load_start, next_exe_start_times[best_num_load]-last_load_remain)
                    offset = new_last_load_start - last_load_start
                    new_last_load_end = last_load_end + offset
                    load_list[last_load_i] = (new_last_load_start, new_last_load_end)
                    for i in range(last_load_i+1, len(load_list)):
                        load_start, load_end = load_list[i]
                        load_list[i] = (load_start+offset, load_end+offset)

            # calculate the hbm time required to load the next op
            next_load_size_byte = self.ops[next_op_idx].min_cold_size_bytes_per_core * np.prod(self.num_cores)
            next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
            next_load_time_cycle = next_load_time_ms * 1.325e6

            # update load time due to noc
            if self.ops[next_op_idx].min_cold_size_bytes_per_core > SKIP_OP_THRESHOLD:
                if MESH:
                    next_load_noc_cycle = next_load_time_cycle/MESH
                else:
                    broadcast_ratio = cold / self.ops[next_op_idx].min_cold_size_bytes_per_core
                    next_load_noc_cycle = next_load_time_cycle * broadcast_ratio / noc_hbm_bw_ratio
                    next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle/load_noc_threshold)
            else:
                next_load_noc_cycle = 0

            # determine end time of next load
            cur_load_start, cur_load_end = load_list[-1]
            next_load_end = min(next_exe_start_times[best_num_load], cur_load_start)

            exe_list.append((next_exe_start_times[best_num_load], next_exe_deadlines[best_num_load]))
            load_list.append((next_load_end-next_load_time_cycle, next_load_end))
            num_overlap_load.append(len(op_group_plans[best_num_load])-1)
            cold_hot_list.append((cold, hot))

            comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][2]
            comp_cycles = comp_cycles/COMP
            shift_cycles = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][3]
            shift_cycles = shift_cycles/COMM
            comp_shift_list.append((comp_cycles, shift_cycles))

            load_remain_noc_list.append(next_load_time_cycle-next_load_noc_cycle)
        
        # assert len(exe_list) == len(self.ops), "length of exe_list should be equal to length of self.ops"
        # assert len(load_list) == len(self.ops), "length of load_list should be equal to length of self.ops"
        # assert len(cold_hot_list) == len(self.ops), "length of cold_hot_list should be equal to length of self.ops"
        # assert len(comp_shift_list) == len(self.ops), "length of comp_shift_list should be equal to length of self.ops"
        # assert len(num_overlap_load) == len(self.ops), "length of num_overlap_load should be equal to length of self.ops"
        # assert len(load_remain_noc_list) == len(self.ops), "length of load_remain_noc_list should be equal to length of self.ops"
        
        exe_list.reverse()
        load_list.reverse()
        cold_hot_list.reverse()
        comp_shift_list.reverse()
        num_overlap_load.reverse()
        load_remain_noc_list.reverse()

        return exe_list, load_list, cold_hot_list, comp_shift_list, num_overlap_load, load_remain_noc_list


    def baseline_search_optimal_exe_load_config_all(self, hbm_GBps: float, num_layers: int, 
                                                    exe_kb_per_core: int, use_largest_cold: bool = False,
                                                    delay_comp: bool = DELAY_COMP_BASE) \
        -> Tuple[ List[Tuple[float,float]],
                  List[Tuple[float,float]],
                  List[Tuple[int  ,int  ]],
                  List[Tuple[float,float]],
                  List[int], List[float] ]:

        exe_byte_per_core = exe_kb_per_core * 1024
        load_byte_per_core = self.tot_mem_size_per_core - exe_byte_per_core
        max_load = len(self.ops) // num_layers  # never pre-load more than a layer of ops
        noc_hbm_bw_ratio = (int(GBPS_PER_CORE)*self.tot_num_cores) / hbm_GBps

        # exe time of the last op
        next_cold, next_hot = self.baseline_cold_hot(self.ops[-1], exe_byte_per_core, use_largest_cold)
        next_exe_time_cycle:float = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][2]
        comp_cycles = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][1][1][2]
        shift_cycles = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][1][1][3]
        new_shift_cycles = shift_cycles/COMM
        next_exe_time_cycle -= (shift_cycles-new_shift_cycles)
        shift_cycles = new_shift_cycles
        new_comp_cycles = comp_cycles/COMP
        next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
        comp_cycles = new_comp_cycles

        # hbm latency of last op
        next_load_size_byte = self.ops[-1].min_cold_size_bytes_per_core * self.tot_num_cores
        next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
        next_load_time_cycle = next_load_time_ms * 1.325e6

        # calculate the noc time required to deliver data to cores, when the last op is loading
        if self.ops[-1].min_cold_size_bytes_per_core > SKIP_OP_THRESHOLD:
            if MESH:
                next_load_noc_cycle = next_load_time_cycle/MESH
            else:
                broadcast_ratio = next_cold / self.ops[-1].min_cold_size_bytes_per_core
                next_load_noc_cycle = next_load_time_cycle * broadcast_ratio / noc_hbm_bw_ratio
                next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle/LOAD_NOC_THRESHOLD_BASE)
        else:
            next_load_noc_cycle = 0
        # real load time is the max of hbm load time and required noc time

        # initialize the lists using the last op's only possible plan
        exe_list = [(-next_exe_time_cycle, 0.)]
        load_list = [(-next_exe_time_cycle-next_load_time_cycle, -next_exe_time_cycle)]
        num_overlap_load = [0]
        cold_hot_list: List[Tuple[int, int]] = [(next_cold, next_hot)]
        comp_shift_list = [(comp_cycles, shift_cycles)]

        # the remaining noc time that can be used for shifting
        load_remain_noc_list = [next_load_time_cycle-next_load_noc_cycle]

        while len(exe_list) < len(self.ops):
            next_op_idx = len(self.ops) - len(exe_list) - 1
            next_cold, next_hot = self.baseline_cold_hot(self.ops[next_op_idx], exe_byte_per_core, use_largest_cold)
            if MESH>=1 and MESH<=1.15:
                rate = MESH**10
                delay_comp = next_op_idx%int(10//rate)==0

            next_exe_deadline = -1
            op_group:List[int] = []
            cur_exe_start, cur_exe_end = exe_list[-1]
            load_byte_demand = 0
            # find all possible deadlines for the exe of next op, and corresponding ops to overlap with
            for load_i in range(min(len(load_list), max_load)):
                rev_load_i = len(load_list) - 1 - load_i
                op_i = next_op_idx + load_i + 1
                op_group.append(op_i-1)
                cur_load_start, cur_load_end = load_list[rev_load_i]
                load_cold, load_hot = cold_hot_list[rev_load_i]
                
                if cur_load_start >= cur_exe_start:
                    next_exe_deadline = cur_exe_start
                    break
                elif load_byte_demand + load_cold > load_byte_per_core:
                    next_exe_deadline = cur_load_start
                    break
                else:
                    load_byte_demand += load_cold
                    next_exe_deadline = cur_load_start

            # get the default start time of exe
            next_exe_time_cycle = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][2]

            shift_cycles_to_handle = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][3]
            new_shift_cycles = shift_cycles_to_handle/COMM
            next_exe_time_cycle -= (shift_cycles_to_handle-new_shift_cycles)
            shift_cycles_to_handle = new_shift_cycles

            comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][2]
            new_comp_cycles = comp_cycles/COMP
            next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
            comp_cycles = new_comp_cycles

            next_exe_start_time = next_exe_deadline - next_exe_time_cycle

            if delay_comp:
                # get the noc time required for exe next op
                # shift_cycles_to_handle = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][3]
                # get number of preloaded ops
                num_load_ops = len(op_group)-1
                next_exe_start_time_noc = 1
                for load_i in range(num_load_ops):
                    # start from the lastest overlapped op
                    rev_load_i_ = len(load_list) - num_load_ops + load_i
                    # remaining noc time reserved for shifting
                    noc_cycle = load_remain_noc_list[rev_load_i_]
                    # end time of the remaining noc time
                    load_start, load_end = load_list[rev_load_i_]
                    # update the two if the load is partially overlapped with the exe
                    if load_end > next_exe_deadline:
                        adjusted_noc_cycle = noc_cycle * (next_exe_deadline - load_start) / (load_end - load_start)
                        assert adjusted_noc_cycle >= 0, "adjusted_noc_cycle should be non-negative"
                        adjusted_noc_end = next_exe_deadline
                    else:
                        adjusted_noc_cycle = noc_cycle
                        adjusted_noc_end = load_end
                    # fit in shift to available noc time
                    shift_cycles_to_handle -= adjusted_noc_cycle
                    # get the latest start time of the exe of next op, considering the noc contention
                    if shift_cycles_to_handle <= 0:
                        shift_cycles_to_handle += adjusted_noc_cycle
                        adjusted_load_duration = adjusted_noc_end-load_start
                        adjusted_shift_cycles = adjusted_load_duration * (shift_cycles_to_handle/adjusted_noc_cycle)
                        next_exe_start_time_noc = adjusted_noc_end-adjusted_shift_cycles
                        break
                # if shift cannot fit in existing loads, start before the beginning of the last load
                if next_exe_start_time_noc == 1:
                    next_exe_start_time_noc = load_list[-1][0]-shift_cycles_to_handle
                next_exe_start_time = min(next_exe_start_time, next_exe_start_time_noc)

            else:
                # post-process the load time
                comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][2]
                comp_cycles = comp_cycles/COMP
                num_load_ops = len(op_group)-1
                last_load_i = -1
                last_load_remain = -1
                for load_i in range(num_load_ops):
                    rev_load_i_ = len(load_list) - num_load_ops + load_i
                    load_start, load_end = load_list[rev_load_i_]
                    load_duration = load_end - load_start
                    noc_cycle = load_duration - load_remain_noc_list[rev_load_i_]
                    if load_end > next_exe_deadline:
                        adjusted_noc_end = next_exe_deadline
                        adjusted_noc_duration = adjusted_noc_end - load_start
                        if noc_cycle == 0:
                            adjusted_noc_cycle = 0
                        else:
                            adjusted_noc_cycle = noc_cycle * adjusted_noc_duration / load_duration
                    else:
                        adjusted_noc_cycle = noc_cycle
                        adjusted_noc_end = load_end
                        adjusted_noc_duration = load_duration
                    comp_cycles -= adjusted_noc_cycle
                    if comp_cycles <= 0:
                        last_load_i = rev_load_i_
                        last_load_remain = (-comp_cycles) * (load_duration/noc_cycle)
                        break
                    if load_start < next_exe_start_time:
                        break
                if last_load_i != -1:
                    last_load_start, last_load_end = load_list[last_load_i]
                    new_last_load_start = min(last_load_start, next_exe_start_time-last_load_remain)
                    offset = new_last_load_start - last_load_start
                    new_last_load_end = last_load_end + offset
                    load_list[last_load_i] = (new_last_load_start, new_last_load_end)
                    for i in range(last_load_i+1, len(load_list)):
                        load_start, load_end = load_list[i]
                        load_list[i] = (load_start+offset, load_end+offset)

            # calculate the hbm time required to load the next op
            next_load_size_byte = self.ops[next_op_idx].min_cold_size_bytes_per_core * self.tot_num_cores
            next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
            next_load_time_cycle = next_load_time_ms * 1.325e6

            # update load time due to noc
            if self.ops[next_op_idx].min_cold_size_bytes_per_core > SKIP_OP_THRESHOLD:
                if MESH:
                    next_load_noc_cycle = next_load_time_cycle/MESH
                else:
                    broadcast_ratio = next_cold / self.ops[next_op_idx].min_cold_size_bytes_per_core
                    next_load_noc_cycle = next_load_time_cycle * broadcast_ratio / noc_hbm_bw_ratio
                    next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle/LOAD_NOC_THRESHOLD_BASE)
            else:
                next_load_noc_cycle = 0

            # determine end time of next load
            cur_load_start, cur_load_end = load_list[-1]
            next_load_end = min(next_exe_start_time, cur_load_start)

            exe_list.append((next_exe_start_time, next_exe_deadline))
            load_list.append((next_load_end-next_load_time_cycle, next_load_end))
            num_overlap_load.append(len(op_group)-1)
            cold_hot_list.append((next_cold, next_hot))

            comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][2]
            comp_cycles = comp_cycles/COMP
            shift_cycles = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][3]
            shift_cycles = shift_cycles/COMM
            comp_shift_list.append((comp_cycles, shift_cycles))

            load_remain_noc_list.append(next_load_time_cycle-next_load_noc_cycle)

        # assert len(exe_list) == len(self.ops), "length of exe_list should be equal to length of self.ops"
        # assert len(load_list) == len(self.ops), "length of load_list should be equal to length of self.ops"
        # assert len(cold_hot_list) == len(self.ops), "length of cold_hot_list should be equal to length of self.ops"
        # assert len(comp_shift_list) == len(self.ops), "length of comp_shift_list should be equal to length of self.ops"
        # assert len(num_overlap_load) == len(self.ops), "length of num_overlap_load should be equal to length of self.ops"
        # assert len(load_remain_noc_list) == len(self.ops), "length of load_remain_noc_list should be equal to length of self.ops"

        exe_list.reverse()
        load_list.reverse()
        cold_hot_list.reverse()
        comp_shift_list.reverse()
        num_overlap_load.reverse()
        load_remain_noc_list.reverse()

        return exe_list, load_list, cold_hot_list, comp_shift_list, num_overlap_load, load_remain_noc_list
    
    # True, False, pick order
    def search_optimal_exe_load_config_order(self, hbm_GBps: float, num_layers: int,
                                             order: List[int] = [], delay_comp: bool = DELAY_COMP_REORDER) \
        -> Tuple[ List[Tuple[float,float]], 
                  List[Tuple[float,float]], 
                  List[Tuple[int  ,int  ]],
                  List[Tuple[float,float]],
                  List[int], List[float] ]:
                
        # default order is the order of ops
        if len(order) == 0:
            order = list(range(len(self.ops)))
            is_default_order = True
        elif order == list(range(len(self.ops))):
            is_default_order = True
        else:
            is_default_order = False
        
        len_self_ops = len(self.ops)
        assert len(order) == len_self_ops, "length of order should be equal to the number of ops"
        op_idx_to_load_order = {op_idx: i for i, op_idx in enumerate(order)}

        # key: op_idx, value: (0: load time cycle, 1: remaining noc time cycle)
        load_info_dict: Dict[int, Tuple[float, float]] = {}

        # self.init_min_cold_size_bytes_per_core()
        plan_dict = {}  # {op_group_keys: op_group_plan} used to cache the search result for each op group
        max_load = len_self_ops // num_layers  # never pre-load more than a layer of ops
        noc_hbm_bw_ratio = (GBPS_PER_CORE*self.tot_num_cores) / hbm_GBps

        # best exe plan for the last op
        op_group_plans = self.search_optimal_global_config_icbm([self.ops[-1:]])
        assert len(op_group_plans) == 1, "length of op_group_plans should be 1"

        # exe time of the last op
        cold, hot = op_group_plans[0][0][0]
        plan = self.ops[-1].expr.cold_hot_table[cold][hot]
        next_exe_time_cycle:float = plan[2]
        comp_cycles = plan[1][1][2]
        shift_cycles = plan[1][1][3]
        new_shift_cycles = shift_cycles/COMM
        next_exe_time_cycle -= (shift_cycles-new_shift_cycles)
        shift_cycles = new_shift_cycles
        new_comp_cycles = comp_cycles/COMP
        next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
        comp_cycles = new_comp_cycles

        # hbm latency of last op
        next_load_size_byte = self.ops[-1].min_cold_size_bytes_per_core * self.tot_num_cores
        next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
        next_load_time_cycle = next_load_time_ms * 1.325e6
        
        # calculate the noc time required to deliver data to cores, when the last op is loading
        if self.ops[-1].min_cold_size_bytes_per_core > SKIP_OP_THRESHOLD:
            if MESH:
                next_load_noc_cycle = next_load_time_cycle/MESH
            else:
                broadcast_ratio = cold / self.ops[-1].min_cold_size_bytes_per_core
                next_load_noc_cycle = next_load_time_cycle * broadcast_ratio / noc_hbm_bw_ratio
                next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle/LOAD_NOC_THRESHOLD_REORDER)
        else:
            next_load_noc_cycle = 0
        # real load time is the max of hbm load time and required noc time

        # initialize the lists using the last op's only possible plan
        exe_list = [(-next_exe_time_cycle, 0.)]
        # load_list = [(-next_exe_time_cycle-next_load_time_cycle, -next_exe_time_cycle)]
        load_info_dict[len_self_ops-1] = (next_load_time_cycle, next_load_time_cycle-next_load_noc_cycle)
        num_overlap_load = [0]
        cold_hot_list = [(cold, hot)]
        comp_shift_list = [(comp_cycles, shift_cycles)]

        # update the load time, and the remaining noc time that can be used for shifting
        if order[-1] == len_self_ops-1:
            load_list = [(-next_exe_time_cycle-next_load_time_cycle, -next_exe_time_cycle)]
            load_remain_noc_list = [next_load_time_cycle-next_load_noc_cycle]
            load_op_idx_list = [len_self_ops-1]
        else:
            load_list = []
            load_remain_noc_list = []
            load_op_idx_list = []

        while len(exe_list) < len_self_ops:
            next_op_idx = len_self_ops - len(exe_list) - 1
            
            next_exe_deadlines = []
            op_group_list:List[List[TensorOperator]] = []
            cur_exe_start, cur_exe_end = exe_list[-1]
            # find all possible deadlines for the exe of next op, and corresponding ops to overlap with
            cur_op_group: List[int] = [next_op_idx]
            load_order_idx = op_idx_to_load_order[next_op_idx]
            load_order_start_idx = max(0, load_order_idx-max_load)
            prefix_order = np.array(order[load_order_start_idx:op_idx_to_load_order[next_op_idx]])
            cur_op_group += prefix_order[prefix_order > next_op_idx].tolist()
            num_uncommited_load = len(cur_op_group)-1

            cur_max_load = max_load-len(cur_op_group)+1
            len_load_list = len(load_list)
            for load_i in range(min(len_load_list, cur_max_load)):
                # give up when extremely memory bounded
                if is_default_order == False and load_i == cur_max_load-1:
                    return [], [], [], [], [], []

                rev_load_i = len_load_list - 1 - load_i
                op_i = load_op_idx_list[rev_load_i]
                cur_load_start, cur_load_end = load_list[rev_load_i]
                
                if cur_load_start >= cur_exe_start:
                    next_exe_deadlines.append(cur_exe_start)
                    op_group_list.append([self.ops[idx] for idx in cur_op_group])
                    break
                elif self.ops[op_i].min_cold_size_bytes_per_core < SKIP_OP_THRESHOLD:
                    cur_op_group.append(op_i)
                    continue
                else:
                    next_exe_deadlines.append(cur_load_start)
                    op_group_list.append([self.ops[idx] for idx in cur_op_group])
                    cur_op_group.append(op_i)

            # if the last op is not loading data, it is small and elementwise, so we allow it to start late and overlap more
            if self.ops[next_op_idx].min_cold_size_bytes_per_core == 0:
                next_exe_deadlines = next_exe_deadlines[-1:]
                op_group_list = op_group_list[-1:]

            # use inter-op optimization to find the best plan for each op overlap plan
            # each op overlap plan is a op group list [exe_op, load_op1, load_op2, ...]
            op_group_plans = []
            for op_group in op_group_list:
                op_group_keys = tuple([op.expr.log_filename_physical for op in op_group])
                # use cached result if possible
                if op_group_keys in plan_dict:
                    op_group_plan = plan_dict[op_group_keys]
                else:
                    op_group_plan = self.search_optimal_global_config_icbm([op_group])
                    plan_dict[op_group_keys] = op_group_plan
                if op_group_plan == []:
                    break
                else:
                    op_group_plans += op_group_plan

            # remove op groups that have no valid plan
            if len(op_group_plans) == 0:
                # print(f"Out of memory, invalid order.")
                return [], [], [], [], [], []
            next_exe_deadlines = next_exe_deadlines[:len(op_group_plans)]

            # find the real exe time of each plan, with the noc contention considered
            next_exe_start_times = []
            cold_hot_plans = []
            for op_group, deadline in zip(op_group_plans, next_exe_deadlines):
                cold, hot = op_group[0][0]
                cold_hot_plans.append((cold, hot))

                # get the default start time of exe
                plan = self.ops[next_op_idx].expr.cold_hot_table[cold][hot]
                next_exe_time_cycle = plan[2]
                shift_cycles_to_handle = plan[1][1][3]
                comp_cycles = plan[1][1][2]

                new_shift_cycles = shift_cycles_to_handle/COMM
                next_exe_time_cycle -= (shift_cycles_to_handle-new_shift_cycles)
                shift_cycles_to_handle = new_shift_cycles

                new_comp_cycles = comp_cycles/COMP
                next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
                comp_cycles = new_comp_cycles

                next_exe_start_time = deadline - next_exe_time_cycle

                if delay_comp:
                    # get the noc time required for exe next op
                    # shift_cycles_to_handle = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][3]
                    # get number of preloaded ops
                    num_load_ops = len(op_group[0])-1-num_uncommited_load
                    next_exe_start_time_noc = 1
                    for load_i in range(num_load_ops):
                        # start from the lastest overlapped op
                        rev_load_i_ = len_load_list - num_load_ops + load_i
                        # remaining noc time reserved for shifting
                        noc_cycle = load_remain_noc_list[rev_load_i_]
                        # end time of the remaining noc time
                        load_start, load_end = load_list[rev_load_i_]
                        # update the two if the load is partially overlapped with the exe
                        if load_end > deadline:
                            adjusted_noc_cycle = noc_cycle * (deadline - load_start) / (load_end - load_start)
                            assert adjusted_noc_cycle >= 0, "adjusted_noc_cycle should be non-negative"
                            adjusted_noc_end = deadline
                        else:
                            adjusted_noc_cycle = noc_cycle
                            adjusted_noc_end = load_end
                        # fit in shift to available noc time
                        shift_cycles_to_handle -= adjusted_noc_cycle
                        # get the latest start time of the exe of next op, considering the noc contention
                        if shift_cycles_to_handle <= 0:
                            shift_cycles_to_handle += adjusted_noc_cycle
                            adjusted_load_duration = adjusted_noc_end-load_start
                            adjusted_shift_cycles = adjusted_load_duration * (shift_cycles_to_handle/adjusted_noc_cycle)
                            next_exe_start_time_noc = adjusted_noc_end-adjusted_shift_cycles
                            break
                    # if shift cannot fit in existing loads, start before the beginning of the last load
                    if next_exe_start_time_noc == 1:
                        next_exe_start_time_noc = load_list[-1][0]-shift_cycles_to_handle
                    next_exe_start_times.append(min(next_exe_start_time, next_exe_start_time_noc))
                else:
                    next_exe_start_times.append(next_exe_start_time)

            # pick latest start time for the exe of next op
            best_num_load = np.argmax(next_exe_start_times)
            cold, hot = cold_hot_plans[best_num_load]

            if not delay_comp:
                # post-process the load time
                comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][2]
                comp_cycles = comp_cycles/COMP
                num_load_ops = len(op_group_plans[best_num_load][0])-1-num_uncommited_load
                last_load_i = -1
                last_load_remain = -1
                for load_i in range(num_load_ops):
                    rev_load_i_ = len(load_list) - num_load_ops + load_i
                    load_start, load_end = load_list[rev_load_i_]
                    load_duration = load_end - load_start
                    noc_cycle = load_duration - load_remain_noc_list[rev_load_i_]
                    if load_end > next_exe_deadlines[best_num_load]:
                        adjusted_noc_end = next_exe_deadlines[best_num_load]
                        adjusted_noc_duration = adjusted_noc_end - load_start
                        if noc_cycle == 0:
                            adjusted_noc_cycle = 0
                        else:
                            adjusted_noc_cycle = noc_cycle * adjusted_noc_duration / load_duration
                    else:
                        adjusted_noc_cycle = noc_cycle
                        adjusted_noc_end = load_end
                        adjusted_noc_duration = load_duration
                    comp_cycles -= adjusted_noc_cycle
                    if comp_cycles <= 0:
                        last_load_i = rev_load_i_
                        last_load_remain = (-comp_cycles) * (load_duration/noc_cycle)
                        break
                    if load_start < next_exe_start_times[best_num_load]:
                        break
                if last_load_i != -1:
                    last_load_start, last_load_end = load_list[last_load_i]
                    new_last_load_start = min(last_load_start, next_exe_start_times[best_num_load]-last_load_remain)
                    offset = new_last_load_start - last_load_start
                    new_last_load_end = last_load_end + offset
                    load_list[last_load_i] = (new_last_load_start, new_last_load_end)
                    for i in range(last_load_i+1, len(load_list)):
                        load_start, load_end = load_list[i]
                        load_list[i] = (load_start+offset, load_end+offset)

            # calculate the hbm time required to load the next op
            next_load_size_byte = self.ops[next_op_idx].min_cold_size_bytes_per_core * self.tot_num_cores
            next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
            next_load_time_cycle = next_load_time_ms * 1.325e6

            # update load time due to noc
            if self.ops[next_op_idx].min_cold_size_bytes_per_core > SKIP_OP_THRESHOLD:
                if MESH:
                    next_load_noc_cycle = next_load_time_cycle/MESH
                else:
                    broadcast_ratio = cold / self.ops[next_op_idx].min_cold_size_bytes_per_core
                    next_load_noc_cycle = next_load_time_cycle * broadcast_ratio / noc_hbm_bw_ratio
                    next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle/LOAD_NOC_THRESHOLD_REORDER)
            else:
                next_load_noc_cycle = 0

            # # determine end time of next load
            # cur_load_start, cur_load_end = load_list[-1]
            # next_load_end = min(next_exe_start_times[best_num_load], cur_load_start)

            exe_list.append((next_exe_start_times[best_num_load], next_exe_deadlines[best_num_load]))
            # load_list.append((next_load_end-next_load_time_cycle, next_load_end))
            load_info_dict[next_op_idx] = (next_load_time_cycle, next_load_time_cycle-next_load_noc_cycle)
            num_overlap_load.append(len(op_group_plans[best_num_load])-1)
            cold_hot_list.append((cold, hot))

            comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][2]
            shift_cycles = self.ops[next_op_idx].expr.cold_hot_table[cold][hot][1][1][3]
            comp_cycles = comp_cycles/COMP
            shift_cycles = shift_cycles/COMM
            comp_shift_list.append((comp_cycles, shift_cycles))

            cur_order_idx = len_self_ops - len(load_list) - 1
            while cur_order_idx >= 0 and order[cur_order_idx] >= next_op_idx:
                next_load_start_time, next_load_end_time = load_list[-1]

                op_id = order[cur_order_idx]
                load_time_cycle, load_remain_noc_cycle = load_info_dict[op_id]

                rev_exe_i = len_self_ops - 1 - op_id
                its_exe_start_time = exe_list[rev_exe_i][0]

                its_load_end_time = min(its_exe_start_time, next_load_start_time)

                load_list.append((its_load_end_time-load_time_cycle, its_load_end_time))
                load_remain_noc_list.append(load_remain_noc_cycle)
                load_op_idx_list.append(op_id)

                cur_order_idx -= 1

        # assert len(exe_list) == len_self_ops, "length of exe_list should be equal to length of self.ops"
        assert len(load_list) == len_self_ops, "length of load_list should be equal to length of self.ops"
        # assert len(cold_hot_list) == len_self_ops, "length of cold_hot_list should be equal to length of self.ops"
        # assert len(comp_shift_list) == len_self_ops, "length of comp_shift_list should be equal to length of self.ops"
        # assert len(num_overlap_load) == len_self_ops, "length of num_overlap_load should be equal to length of self.ops"
        # assert len(load_remain_noc_list) == len_self_ops, "length of load_remain_noc_list should be equal to length of self.ops"
        # assert len(load_op_idx_list) == len_self_ops, "length of load_op_idx_list should be equal to length of self.ops"
        
        exe_list.reverse()
        load_list.reverse()
        cold_hot_list.reverse()
        comp_shift_list.reverse()
        num_overlap_load.reverse()
        load_remain_noc_list.reverse()

        return exe_list, load_list, cold_hot_list, comp_shift_list, num_overlap_load, load_remain_noc_list
    

    def ideal_search_optimal_exe_load_config_all(self, hbm_GBps: float, num_layers: int) \
        -> Tuple[ List[Tuple[float,float]],
                  List[Tuple[float,float]],
                  List[Tuple[int  ,int  ]],
                  List[Tuple[float,float]],
                  List[int], List[float] ]:

        max_load = len(self.ops) // num_layers  # never pre-load more than a layer of ops
        noc_hbm_bw_ratio = (GBPS_PER_CORE*self.tot_num_cores) / hbm_GBps

        # exe time of the last op
        next_plan = self.search_optimal_global_config_icbm([[self.ops[-1]]], ideal=True)
        next_cold, next_hot = next_plan[0][0][0]
        next_exe_time_cycle:float = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][2]
        comp_cycles = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][1][1][2]
        shift_cycles = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][1][1][3]
        new_shift_cycles = shift_cycles/COMM
        next_exe_time_cycle -= (shift_cycles-new_shift_cycles)
        shift_cycles = new_shift_cycles
        new_comp_cycles = comp_cycles/COMP
        next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
        comp_cycles = new_comp_cycles

        # hbm latency of last op
        next_load_size_byte = self.ops[-1].min_cold_size_bytes_per_core * self.tot_num_cores
        next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
        next_load_time_cycle = next_load_time_ms * 1.325e6

        # calculate the noc time required to deliver data to cores, when the last op is loading
        next_load_noc_cycle = next_load_time_cycle / noc_hbm_bw_ratio

        # real load time is the max of hbm load time and required noc time
        next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle)

        # initialize the lists using the last op's only possible plan
        exe_list = [(-next_exe_time_cycle, 0.)]
        load_list = [(-next_exe_time_cycle-next_load_time_cycle, -next_exe_time_cycle)]
        num_overlap_load = [0]
        cold_hot_list: List[Tuple[int, int]] = [(next_cold, next_hot)]
        comp_shift_list = [(comp_cycles, shift_cycles)]

        # the remaining noc time that can be used for shifting
        load_remain_noc_list = [next_load_time_cycle-next_load_noc_cycle]

        while len(exe_list) < len(self.ops):
            next_op_idx = len(self.ops) - len(exe_list) - 1
            next_plan = self.search_optimal_global_config_icbm([[self.ops[next_op_idx]]], ideal=True)
            next_cold, next_hot = next_plan[0][0][0]

            min_cold = next(iter(self.ops[next_op_idx].expr.cold_hot_table))
            min_hot = next(iter(self.ops[next_op_idx].expr.cold_hot_table[min_cold]))
            load_byte_per_core = self.tot_mem_size_per_core - min_hot

            next_exe_deadline = -1
            op_group:List[int] = []
            cur_exe_start, cur_exe_end = exe_list[-1]
            load_byte_demand = 0
            # find all possible deadlines for the exe of next op, and corresponding ops to overlap with
            for load_i in range(min(len(load_list), max_load)):
                rev_load_i = len(load_list) - 1 - load_i
                op_i = next_op_idx + load_i + 1
                op_group.append(op_i-1)
                cur_load_start, cur_load_end = load_list[rev_load_i]
                load_cold = self.ops[op_i].min_cold_size_bytes_per_core
                
                if cur_load_start >= cur_exe_start:
                    next_exe_deadline = cur_exe_start
                    break
                elif load_byte_demand + load_cold > load_byte_per_core:
                    next_exe_deadline = cur_load_start
                    break
                else:
                    load_byte_demand += load_cold
                    next_exe_deadline = cur_load_start

            # get the default start time of exe
            next_exe_time_cycle = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][2]
            
            shift_cycles_to_handle = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][3]
            new_shift_cycles = shift_cycles_to_handle/COMM
            next_exe_time_cycle -= (shift_cycles_to_handle-new_shift_cycles)
            shift_cycles_to_handle = new_shift_cycles
            
            comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][2]
            new_comp_cycles = comp_cycles/COMP
            next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
            comp_cycles = new_comp_cycles

            next_exe_start_time = next_exe_deadline - next_exe_time_cycle

            # calculate the hbm time required to load the next op
            next_load_size_byte = self.ops[next_op_idx].min_cold_size_bytes_per_core * self.tot_num_cores
            next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
            next_load_time_cycle = next_load_time_ms * 1.325e6

            # update load time due to noc
            next_load_noc_cycle = next_load_time_cycle / noc_hbm_bw_ratio
            next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle)

            # determine end time of next load
            cur_load_start, cur_load_end = load_list[-1]
            next_load_end = min(next_exe_start_time, cur_load_start)

            exe_list.append((next_exe_start_time, next_exe_deadline))
            load_list.append((next_load_end-next_load_time_cycle, next_load_end))
            num_overlap_load.append(len(op_group)-1)
            cold_hot_list.append((next_cold, next_hot))

            shift_cycles = shift_cycles_to_handle
            comp_shift_list.append((comp_cycles, shift_cycles))

            load_remain_noc_list.append(next_load_time_cycle-next_load_noc_cycle)

        # assert len(exe_list) == len(self.ops), "length of exe_list should be equal to length of self.ops"
        # assert len(load_list) == len(self.ops), "length of load_list should be equal to length of self.ops"
        # assert len(cold_hot_list) == len(self.ops), "length of cold_hot_list should be equal to length of self.ops"
        # assert len(comp_shift_list) == len(self.ops), "length of comp_shift_list should be equal to length of self.ops"
        # assert len(num_overlap_load) == len(self.ops), "length of num_overlap_load should be equal to length of self.ops"
        # assert len(load_remain_noc_list) == len(self.ops), "length of load_remain_noc_list should be equal to length of self.ops"

        exe_list.reverse()
        load_list.reverse()
        cold_hot_list.reverse()
        comp_shift_list.reverse()
        num_overlap_load.reverse()
        load_remain_noc_list.reverse()

        return exe_list, load_list, cold_hot_list, comp_shift_list, num_overlap_load, load_remain_noc_list
    

    def naive_search_optimal_exe_load_config_all(self, hbm_GBps: float, num_layers: int,
                                                 use_largest_cold: bool = True, delay_comp: bool = DELAY_COMP_NAIVE) \
        -> Tuple[ List[Tuple[float,float]],
                  List[Tuple[float,float]],
                  List[Tuple[int  ,int  ]],
                  List[Tuple[float,float]],
                  List[int], List[float] ]:

        max_load = len(self.ops) // num_layers  # never pre-load more than a layer of ops
        noc_hbm_bw_ratio = (int(GBPS_PER_CORE)*self.tot_num_cores) / hbm_GBps

        # exe time of the last op
        next_cold, next_hot = self.baseline_cold_hot(self.ops[-1], self.tot_mem_size_per_core, 
                                                     use_largest_cold=use_largest_cold)
        next_exe_time_cycle:float = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][2]
        comp_cycles = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][1][1][2]
        shift_cycles = self.ops[-1].expr.cold_hot_table[next_cold][next_hot][1][1][3]
        new_shift_cycles = shift_cycles/COMM
        next_exe_time_cycle -= (shift_cycles-new_shift_cycles)
        shift_cycles = new_shift_cycles
        new_comp_cycles = comp_cycles/COMP
        next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
        comp_cycles = new_comp_cycles

        # hbm latency of last op
        next_load_size_byte = self.ops[-1].min_cold_size_bytes_per_core * self.tot_num_cores
        next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
        next_load_time_cycle = next_load_time_ms * 1.325e6

        # calculate the noc time required to deliver data to cores, when the last op is loading
        if self.ops[-1].min_cold_size_bytes_per_core > SKIP_OP_THRESHOLD:
            if MESH:
                next_load_noc_cycle = next_load_time_cycle/MESH
            else:
                broadcast_ratio = next_cold / self.ops[-1].min_cold_size_bytes_per_core
                next_load_noc_cycle = next_load_time_cycle * broadcast_ratio / noc_hbm_bw_ratio
                next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle/LOAD_NOC_THRESHOLD_NAIVE)
        else:
            next_load_noc_cycle = 0
        # real load time is the max of hbm load time and required noc time

        # initialize the lists using the last op's only possible plan
        exe_list = [(-next_exe_time_cycle, 0.)]
        load_list = [(-next_exe_time_cycle-next_load_time_cycle, -next_exe_time_cycle)]
        num_overlap_load = [0]
        cold_hot_list: List[Tuple[int, int]] = [(next_cold, next_hot)]
        comp_shift_list = [(comp_cycles, shift_cycles)]

        # the remaining noc time that can be used for shifting
        load_remain_noc_list = [next_load_time_cycle-next_load_noc_cycle]

        while len(exe_list) < len(self.ops):
            next_op_idx = len(self.ops) - len(exe_list) - 1
            next_exe_deadline = -1
            op_group:List[int] = []
            cur_exe_start, cur_exe_end = exe_list[-1]

            next_min_cold = next(iter(self.ops[next_op_idx].expr.cold_hot_table))
            next_min_hot = next(iter(self.ops[next_op_idx].expr.cold_hot_table[next_min_cold]))
            
            large_op_is_preloaded = False
            load_byte_demand = 0
            load_byte_per_core = self.tot_mem_size_per_core - next_min_hot
            loaded_sizes = []
            # find all possible deadlines for the exe of next op, and corresponding ops to overlap with
            for load_i in range(min(len(load_list), max_load)):
                rev_load_i = len(load_list) - 1 - load_i
                op_i = next_op_idx + load_i + 1
                op_group.append(op_i-1)
                cur_load_start, cur_load_end = load_list[rev_load_i]
                load_cold, load_hot = cold_hot_list[rev_load_i]
                loaded_sizes.append(load_cold)

                if cur_load_start >= cur_exe_start:
                    next_exe_deadline = cur_exe_start
                    break
                elif large_op_is_preloaded or load_byte_demand + load_cold > load_byte_per_core:
                    next_exe_deadline = cur_load_start
                    break
                else:
                    load_byte_demand += load_cold
                    next_exe_deadline = cur_load_start
                    # if self.ops[op_i].op_type == TE.TensorExpression.OP_TYPE_MATMUL or \
                    #     self.ops[op_i].op_type == TE.TensorExpression.OP_TYPE_CONV:
                    if self.ops[op_i].min_cold_size_bytes_per_core > 0:
                        large_op_is_preloaded = True

            loaded_size = sum(loaded_sizes[:len(op_group)-1])
            next_cold, next_hot = self.baseline_cold_hot(self.ops[next_op_idx], self.tot_mem_size_per_core-loaded_size, 
                                                         use_largest_cold=use_largest_cold)

            # get the default start time of exe
            next_exe_time_cycle = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][2]

            shift_cycles_to_handle = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][3]
            new_shift_cycles = shift_cycles_to_handle/COMM
            next_exe_time_cycle -= (shift_cycles_to_handle-new_shift_cycles)
            shift_cycles_to_handle = new_shift_cycles

            comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][2]
            new_comp_cycles = comp_cycles/COMP
            next_exe_time_cycle -= (comp_cycles-new_comp_cycles)
            comp_cycles = new_comp_cycles

            next_exe_start_time = next_exe_deadline - next_exe_time_cycle

            if delay_comp:
                # get the noc time required for exe next op
                # shift_cycles_to_handle = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][3]
                # get number of preloaded ops
                num_load_ops = len(op_group)-1
                next_exe_start_time_noc = 1
                for load_i in range(num_load_ops):
                    # start from the lastest overlapped op
                    rev_load_i_ = len(load_list) - num_load_ops + load_i
                    # remaining noc time reserved for shifting
                    noc_cycle = load_remain_noc_list[rev_load_i_]
                    # end time of the remaining noc time
                    load_start, load_end = load_list[rev_load_i_]
                    # update the two if the load is partially overlapped with the exe
                    if load_end > next_exe_deadline:
                        adjusted_noc_cycle = noc_cycle * (next_exe_deadline - load_start) / (load_end - load_start)
                        assert adjusted_noc_cycle >= 0, "adjusted_noc_cycle should be non-negative"
                        adjusted_noc_end = next_exe_deadline
                    else:
                        adjusted_noc_cycle = noc_cycle
                        adjusted_noc_end = load_end
                    # fit in shift to available noc time
                    shift_cycles_to_handle -= adjusted_noc_cycle
                    # get the latest start time of the exe of next op, considering the noc contention
                    if shift_cycles_to_handle <= 0:
                        shift_cycles_to_handle += adjusted_noc_cycle
                        adjusted_load_duration = adjusted_noc_end-load_start
                        adjusted_shift_cycles = adjusted_load_duration * (shift_cycles_to_handle/adjusted_noc_cycle)
                        next_exe_start_time_noc = adjusted_noc_end-adjusted_shift_cycles
                        break
                # if shift cannot fit in existing loads, start before the beginning of the last load
                if next_exe_start_time_noc == 1:
                    next_exe_start_time_noc = load_list[-1][0]-shift_cycles_to_handle
                next_exe_start_time = min(next_exe_start_time, next_exe_start_time_noc)

            else:
                # post-process the load time
                num_load_ops = len(op_group)-1
                last_load_i = -1
                last_load_remain = -1
                for load_i in range(num_load_ops):
                    rev_load_i_ = len(load_list) - num_load_ops + load_i
                    load_start, load_end = load_list[rev_load_i_]
                    load_duration = load_end - load_start
                    noc_cycle = load_duration - load_remain_noc_list[rev_load_i_]
                    if load_end > next_exe_deadline:
                        adjusted_noc_end = next_exe_deadline
                        adjusted_noc_duration = adjusted_noc_end - load_start
                        if noc_cycle == 0:
                            adjusted_noc_cycle = 0
                        else:
                            adjusted_noc_cycle = noc_cycle * adjusted_noc_duration / load_duration
                    else:
                        adjusted_noc_cycle = noc_cycle
                        adjusted_noc_end = load_end
                        adjusted_noc_duration = load_duration
                    comp_cycles -= adjusted_noc_cycle
                    if comp_cycles <= 0:
                        last_load_i = rev_load_i_
                        last_load_remain = (-comp_cycles) * (load_duration/noc_cycle)
                        break
                    if load_start < next_exe_start_time:
                        break
                if last_load_i != -1:
                    last_load_start, last_load_end = load_list[last_load_i]
                    new_last_load_start = min(last_load_start, next_exe_start_time-last_load_remain)
                    offset = new_last_load_start - last_load_start
                    new_last_load_end = last_load_end + offset
                    load_list[last_load_i] = (new_last_load_start, new_last_load_end)
                    for i in range(last_load_i+1, len(load_list)):
                        load_start, load_end = load_list[i]
                        load_list[i] = (load_start+offset, load_end+offset)

            # calculate the hbm time required to load the next op
            next_load_size_byte = self.ops[next_op_idx].min_cold_size_bytes_per_core * self.tot_num_cores
            next_load_time_ms = float(next_load_size_byte) / 1024. / 1024. / hbm_GBps
            next_load_time_cycle = next_load_time_ms * 1.325e6

            # update load time due to noc
            if self.ops[next_op_idx].min_cold_size_bytes_per_core > SKIP_OP_THRESHOLD:
                if MESH:
                    next_load_noc_cycle = next_load_time_cycle/MESH
                else:
                    broadcast_ratio = next_cold / self.ops[next_op_idx].min_cold_size_bytes_per_core
                    next_load_noc_cycle = next_load_time_cycle * broadcast_ratio / noc_hbm_bw_ratio
                    next_load_time_cycle = max(next_load_time_cycle, next_load_noc_cycle/LOAD_NOC_THRESHOLD_NAIVE)
            else:
                next_load_noc_cycle = 0

            # determine end time of next load
            cur_load_start, cur_load_end = load_list[-1]
            next_load_end = min(next_exe_start_time, cur_load_start)

            exe_list.append((next_exe_start_time, next_exe_deadline))
            load_list.append((next_load_end-next_load_time_cycle, next_load_end))
            num_overlap_load.append(len(op_group)-1)
            cold_hot_list.append((next_cold, next_hot))

            comp_cycles = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][2]
            comp_cycles = comp_cycles/COMP
            shift_cycles = self.ops[next_op_idx].expr.cold_hot_table[next_cold][next_hot][1][1][3]
            shift_cycles = shift_cycles/COMM
            comp_shift_list.append((comp_cycles, shift_cycles))

            load_remain_noc_list.append(next_load_time_cycle-next_load_noc_cycle)

        # assert len(exe_list) == len(self.ops), "length of exe_list should be equal to length of self.ops"
        # assert len(load_list) == len(self.ops), "length of load_list should be equal to length of self.ops"
        # assert len(cold_hot_list) == len(self.ops), "length of cold_hot_list should be equal to length of self.ops"
        # assert len(comp_shift_list) == len(self.ops), "length of comp_shift_list should be equal to length of self.ops"
        # assert len(num_overlap_load) == len(self.ops), "length of num_overlap_load should be equal to length of self.ops"
        # assert len(load_remain_noc_list) == len(self.ops), "length of load_remain_noc_list should be equal to length of self.ops"

        exe_list.reverse()
        load_list.reverse()
        cold_hot_list.reverse()
        comp_shift_list.reverse()
        num_overlap_load.reverse()
        load_remain_noc_list.reverse()

        return exe_list, load_list, cold_hot_list, comp_shift_list, num_overlap_load, load_remain_noc_list
    
        
    def init_min_cold_size_bytes_per_core(self):
        for op in self.ops:
            op.min_cold_size_bytes_per_core = next(iter(op.expr.cold_hot_table))

    
    def init_all_order_lists(self, layer: int):

        all_layer_ops = self.ops
        # all_layer_ops = self.ops[:5] + deepcopy(self.ops) + deepcopy(self.ops)

        approx_layer_len = len(all_layer_ops) // layer # guess the length of each layer

        ss1 = [x.expr.log_filename_physical for x in all_layer_ops[:approx_layer_len]] # chunk first 2 accordingly
        ss2 = [x.expr.log_filename_physical for x in all_layer_ops[approx_layer_len:2*approx_layer_len]]


        # use chunks to eliminate any non-uniform ops from the initial block
        for offset in range(approx_layer_len):
            temp = ss1[offset:]
            for i in range(offset + 1):
                for j in range(approx_layer_len - offset):
                    if ss2[i + j] != ss1[offset:][j]:
                        break
                else:
                    break
            else:
                continue
            break

        actual_layer_len = (approx_layer_len - offset) + i
        print(f"actual_layer_len: {actual_layer_len}")

        all_layer_ops = all_layer_ops[offset:(offset + layer*actual_layer_len)]
        print(len(all_layer_ops), actual_layer_len)
        assert len(all_layer_ops) / layer == actual_layer_len

        # check that we've found the layer
        for i in range(1, layer):
            for a, b in zip(all_layer_ops[:actual_layer_len],
                            all_layer_ops[(i)*actual_layer_len:(i+1)*actual_layer_len]):
                            # all_layer_ops[(layer-1)*actual_layer_len:layer*actual_layer_len]):
                assert a.expr.log_filename_physical == b.expr.log_filename_physical

        layer_ops = all_layer_ops[:actual_layer_len]

        # for op in layer_ops:
            # print(op.expr.log_filename_physical)

        # group layers s.t. there is exactly one 'large' operator per group, others are less relevant
        layer_groups = []
        layer_groups_idx_only = []
        layer_groups_ht = {}


        start_idx = 0
        for i in range(len(layer_ops)):
            if layer_ops[i].op_type == 4 or layer_ops[i].op_type == 5: # CONV or MATMUL
                # layer_groups.append(layer_ops[start_idx:i+1])
                op_dim = layer_ops[i].expr.dim_lengths
                op_size = 1
                for dim in op_dim:
                    if dim != 0:
                        op_size *= dim
                op_name = layer_ops[i].expr.log_filename_physical.split('_')
                op_name = '_'.join([op_name[0], op_name[2]]) + '_' + str(op_size)
                # print(layer_ops[i].expr.log_filename_physical, op_name)
                if op_name not in layer_groups_ht:
                    layer_groups_ht[op_name] = len(layer_groups_ht)

                layer_groups.append(layer_groups_ht[op_name])
                layer_groups_idx_only.append(list(range(offset + start_idx, offset + start_idx + len(layer_ops[start_idx:i+1]))))
                start_idx = i+1
            elif i == len(layer_ops) - 1: # if this is the last, and not major
                layer_groups_idx_only[-1].extend(list(range(offset + start_idx, offset + start_idx + len(layer_ops[start_idx:i+1]))))

        # permute group blocks
        # layer_groups_idx_only = layer_groups_idx_only[:]
        layer_groups_idx_only = layer_groups_idx_only
        layer_groups = layer_groups

        all_group_orderings = list(itertools.permutations(layer_groups_idx_only))
        all_hash_orderings = list(itertools.permutations(layer_groups))

        # flatten each permutation
        all_op_orderings = []
        prev_hash_orderings = []
        orders_eliminated = 0
        for group_ordering, hash_ordering in zip(all_group_orderings, all_hash_orderings):
            hash_string = ""

            for h in hash_ordering:
                hash_string += str(h)
                hash_string += ","

            if hash_string in prev_hash_orderings:
                orders_eliminated += 1
                continue
            else:
                prev_hash_orderings.append(hash_string)

            flat_order = []
            for group in group_ordering:
                flat_order.extend(group)

            assert len(flat_order) == actual_layer_len

            search_order = list(range(offset))
            for i in range(layer):
                search_order += [x + i*len(flat_order) for x in flat_order]

            for i in range(len(self.ops) - len(search_order)):
                search_order.append(len(search_order))

            assert len(search_order) == len(self.ops)
            all_op_orderings.append(search_order)

        print(f"orders_eliminated: {orders_eliminated}")
        print(f"number_of_orders: {len(all_op_orderings)}")

        # self.all_order_lists = all_op_orderings
        return all_op_orderings

    def init_all_order_lists_reduced(self, layer: int, max_edit_dist: int):

        all_layer_ops = self.ops
        # all_layer_ops = self.ops[:5] + deepcopy(self.ops) + deepcopy(self.ops)

        approx_layer_len = len(all_layer_ops) // layer # guess the length of each layer

        ss1 = [x.expr.log_filename_physical for x in all_layer_ops[:approx_layer_len]] # chunk first 2 accordingly
        ss2 = [x.expr.log_filename_physical for x in all_layer_ops[approx_layer_len:2*approx_layer_len]]

        print(f"H: {sum([(1 if (x.op_type == 4 or x.op_type == 5) else 0) for x in all_layer_ops])}", file=sys.stderr)
        # use chunks to eliminate any non-uniform ops from the initial block
        for offset in range(approx_layer_len):
            temp = ss1[offset:]
            for i in range(offset + 1):
                for j in range(approx_layer_len - offset):
                    if ss2[i + j] != ss1[offset:][j]:
                        break
                else:
                    break
            else:
                continue
            break

        actual_layer_len = (approx_layer_len - offset) + i
        print(f"actual_layer_len: {actual_layer_len}", file=sys.stderr)

        all_layer_ops = all_layer_ops[offset:(offset + layer*actual_layer_len)]
        print(f"all_layer_ops: {len(all_layer_ops)}", file=sys.stderr)
        print(f"calc_layer_len: {len(all_layer_ops) / layer}", file=sys.stderr)

        try:
            assert len(all_layer_ops) / layer == actual_layer_len
        except:
            offset = 0
            all_layer_ops = self.ops[:layer*actual_layer_len]
            print(f"failed the assertion", file=sys.stderr)
            print(f"actual_layer_len: {actual_layer_len}", file=sys.stderr)
            # return [list(range(len(self.ops)))]

        # check that we've found the layer
        # for i in range(1, layer):
        #     for a, b in zip(all_layer_ops[:actual_layer_len],
        #                     all_layer_ops[(i)*actual_layer_len:(i+1)*actual_layer_len]):
        #                     # all_layer_ops[(layer-1)*actual_layer_len:layer*actual_layer_len]):
        #         # had to remove this assertion for gemma due to the inserted layers :(
        #         # assert a.expr.log_filename_physical == b.expr.log_filename_physical
        #         if a.expr.log_filename_physical != b.expr.log_filename_physical:
        #             print(i, a.expr.log_filename_physical, b.expr.log_filename_physical, file=sys.stderr)

        layer_ops = all_layer_ops[:actual_layer_len]

        # for op in layer_ops:
            # print(op.expr.log_filename_physical)

        # group layers s.t. there is exactly one 'large' operator per group, others are less relevant
        layer_groups = []
        layer_groups_idx_only = []
        layer_groups_ht = {}

        start_idx = 0
        for i in range(len(layer_ops)):
            if layer_ops[i].op_type == 4 or layer_ops[i].op_type == 5: # CONV or MATMUL
                # layer_groups.append(layer_ops[start_idx:i+1])
                op_dim = layer_ops[i].expr.dim_lengths
                op_size = 1
                for dim in op_dim:
                    if dim != 0:
                        op_size *= dim
                op_name = layer_ops[i].expr.log_filename_physical.split('_')
                op_name = '_'.join([op_name[0], op_name[2]]) + '_' + str(op_size)
                # print(layer_ops[i].expr.log_filename_physical, op_name)
                if op_name not in layer_groups_ht:
                    layer_groups_ht[op_name] = len(layer_groups_ht)

                layer_groups.append(layer_groups_ht[op_name])
                layer_groups_idx_only.append(list(range(offset + start_idx, offset + start_idx + len(layer_ops[start_idx:i+1]))))
                start_idx = i+1
            elif i == len(layer_ops) - 1: # if this is the last, and not major
                layer_groups_idx_only[-1].extend(list(range(offset + start_idx, offset + start_idx + len(layer_ops[start_idx:i+1]))))

        # permute group blocks
        # layer_groups_idx_only = layer_groups_idx_only[:]
        layer_groups_idx_only = layer_groups_idx_only
        layer_groups = layer_groups

        print(layer_groups_idx_only)
        print(layer_groups)

        indexed_perms = reduced_edit_dist_permutations(layer_groups, list(range(len(layer_groups_idx_only))), max_edit_dist)
        print(f"len(indexed_perms): {len(indexed_perms)}")
        # exit(0)

        all_group_orderings = []
        for perm in indexed_perms:
            next_group_ordering = []
            for i in perm:
                next_group_ordering.append(layer_groups_idx_only[i])
            all_group_orderings.append(next_group_ordering)

        print("---------")
        print(indexed_perms[0], file=sys.stderr)
        print(all_group_orderings[0], file=sys.stderr)
        assert len(all_group_orderings[0]) == len(layer_groups_idx_only)

        # flatten each permutation
        all_op_orderings = []
        prev_hash_orderings = []
        orders_eliminated = 0
        for group_ordering in all_group_orderings:
            # hash_string = ""

            # for h in hash_ordering:
            #     hash_string += str(h)
            #     hash_string += ","

            # if hash_string in prev_hash_orderings:
            #     orders_eliminated += 1
            #     continue
            # else:
            #     prev_hash_orderings.append(hash_string)

            flat_order = []
            for group in group_ordering:
                flat_order.extend(group)

            assert len(flat_order) == actual_layer_len

            search_order = list(range(offset))
            for i in range(layer):
                search_order += [x + i*len(flat_order) for x in flat_order]

            for i in range(len(self.ops) - len(search_order)):
                search_order.append(len(search_order))

            assert len(search_order) == len(self.ops)
            all_op_orderings.append(search_order)

        print(f"number_of_orders: {len(all_op_orderings)}", file=sys.stderr)

        # self.all_order_lists = all_op_orderings
        return all_op_orderings

    # def init_all_order_lists_more_reduced(self, layer: int, max_edit_dist: int) -> None:
    #     all_layer_ops = self.ops
    #     # all_layer_ops = self.ops[:5] + deepcopy(self.ops) + deepcopy(self.ops)

    #     approx_layer_len = len(all_layer_ops) // layer # guess the length of each layer

    #     ss1 = [x.expr.log_filename_physical for x in all_layer_ops[:approx_layer_len]] # chunk first 2 accordingly
    #     ss2 = [x.expr.log_filename_physical for x in all_layer_ops[approx_layer_len:2*approx_layer_len]]

    #     # use chunks to eliminate any non-uniform ops from the initial block
    #     for offset in range(approx_layer_len):
    #         temp = ss1[offset:]
    #         for i in range(offset + 1):
    #             for j in range(approx_layer_len - offset):
    #                 if ss2[i + j] != ss1[offset:][j]:
    #                     break
    #             else:
    #                 break
    #         else:
    #             continue
    #         break

    #     actual_layer_len = (approx_layer_len - offset) + i
    #     print(f"actual_layer_len: {actual_layer_len}", file=sys.stderr)

    #     all_layer_ops = all_layer_ops[offset:(offset + layer*actual_layer_len)]
    #     assert len(all_layer_ops) / layer == actual_layer_len

    #     # check that we've found the layer
    #     for i in range(1, layer):
    #         for a, b in zip(all_layer_ops[:actual_layer_len],
    #                         all_layer_ops[(i)*actual_layer_len:(i+1)*actual_layer_len]):
    #                         # all_layer_ops[(layer-1)*actual_layer_len:layer*actual_layer_len]):
    #             assert a.expr.log_filename_physical == b.expr.log_filename_physical

    #     layer_ops = all_layer_ops[:actual_layer_len]

    #     # for op in layer_ops:
    #         # print(op.expr.log_filename_physical)

    #     # group layers s.t. there is exactly one 'large' operator per group, others are less relevant
    #     layer_groups = []
    #     layer_groups_idx_only = []
    #     layer_groups_ht = {}
    #     unique_count = {}

    #     start_idx = 0
    #     for i in range(len(layer_ops)):
    #         if layer_ops[i].op_type == 4 or layer_ops[i].op_type == 5: # CONV or MATMUL
    #             # layer_groups.append(layer_ops[start_idx:i+1])
    #             op_dim = layer_ops[i].expr.dim_lengths
    #             op_size = 1
    #             for dim in op_dim:
    #                 if dim != 0:
    #                     op_size *= dim
    #             op_name = layer_ops[i].expr.log_filename_physical.split('_')
    #             op_name = '_'.join([op_name[0], op_name[2]]) + '_' + str(op_size)
    #             # print(layer_ops[i].expr.log_filename_physical, op_name)
    #             if op_name not in layer_groups_ht:
    #                 layer_groups_ht[op_name] = len(layer_groups_ht)
    #                 unique_count[layer_groups_ht[op_name]] = 1
    #             else:
    #                 unique_count[layer_groups_ht[op_name]] += 1

    #             layer_groups.append(layer_groups_ht[op_name])
    #             layer_groups_idx_only.append(list(range(offset + start_idx, offset + start_idx + len(layer_ops[start_idx:i+1]))))
    #             start_idx = i+1
    #         elif i == len(layer_ops) - 1: # if this is the last, and not major
    #             layer_groups_idx_only[-1].extend(list(range(offset + start_idx, offset + start_idx + len(layer_ops[start_idx:i+1]))))

    #     print(layer_groups)
    #     print(layer_groups_idx_only)
    #     print(layer_groups_ht)
    #     print(unique_count)
    #     hash_to_idx = {}
    #     for i, h in enumerate(layer_groups):
    #         if h not in hash_to_idx:
    #             hash_to_idx[h] = [i]
    #         else:
    #             hash_to_idx[h].append(i)

    #     def recursive_build(cur, blocks, target_len, perm_set):
    #         if (len(cur)/2) == target_len:
    #             perm_set.add(cur)
    #             return

    #         for b in blocks:
    #             if blocks[b] == 0:
    #                 continue
    #             blocks[b] -= 1
    #             recursive_build(cur + "," + str(b), blocks, target_len, perm_set)
    #             blocks[b] += 1

    #     perm_set = set()
    #     recursive_build("", unique_count, len(layer_groups), perm_set)
    #     print(len(perm_set))

    #     indexed_perms = [[int(v) for v in x[1:].split(',')] for x in perm_set]
    #     for g in indexed_perms:
    #         for i, h in enumerate(g):
    #             g[i] = hash_to_idx[h][0]
    #             del hash_to_idx[h][0]
    #             hash_to_idx[h].append(g[i])

    #     # TODO

    #     all_group_orderings = []
    #     for perm in indexed_perms:
    #         next_group_ordering = []
    #         for i in perm:
    #             next_group_ordering.append(layer_groups_idx_only[i])
    #         all_group_orderings.append(next_group_ordering)


    #     all_group_orderings = [all_group_orderings[i] for i in range(0, len(all_group_orderings), 10000)]
    #     print(len(all_group_orderings))

    #     all_op_orderings = []
    #     for group_ordering in all_group_orderings:
    #         flat_order = []
    #         for group in group_ordering:
    #             flat_order.extend(group)
    #         assert len(flat_order) == actual_layer_len

    #         search_order = list(range(offset))
    #         for i in range(layer):
    #             search_order += [x + i*len(flat_order) for x in flat_order]

    #         for i in range(len(self.ops) - len(search_order)):
    #             search_order.append(len(search_order))

    #         all_op_orderings.append(search_order)

    #     # self.all_order_lists = all_op_orderings
    #     return all_op_orderings

        # exit(0)

        # flatten each permutation
        # all_op_orderings = []
        # prev_hash_orderings = []
        # orders_eliminated = 0
        # for group_ordering, hash_ordering in zip(all_group_orderings, all_hash_orderings):
        #     hash_string = ""

        #     for h in hash_ordering:
        #         hash_string += str(h)
        #         hash_string += ","

        #     if hash_string in prev_hash_orderings:
        #         orders_eliminated += 1
        #         continue
        #     else:
        #         prev_hash_orderings.append(hash_string)

        #     flat_order = []
        #     for group in group_ordering:
        #         flat_order.extend(group)

        #     assert len(flat_order) == actual_layer_len

        #     search_order = list(range(offset))
        #     for i in range(layer):
        #         search_order += [x + i*len(flat_order) for x in flat_order]

        #     for i in range(len(self.ops) - len(search_order)):
        #         search_order.append(len(search_order))

        #     assert len(search_order) == len(self.ops)
        #     all_op_orderings.append(search_order)

        # print(f"number_of_orders: {len(all_op_orderings)}", file=sys.stderr)

        # self.all_order_lists = all_op_orderings
