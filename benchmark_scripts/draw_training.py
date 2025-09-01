#!/usr/bin/env python3

from fig_common import *

import matplotlib.pyplot as plt
import numpy as np

import argparse
import DNNProgram as DNNProgram
import pickle



batch_size = 2
kb = 624
seq_length = 2048
num_cores=5888

# hbm_bws = [i for i in range(1000, 5000, 500)] + [i for i in range(5000, 14000, 2000)]
# hbm_bws = [i for i in range(1000, 18000+1, 1000)]
hbm_bws = [300, 400]
noc_bws = [1.0, 1.5]
comp_bws = [0.75, 1.0, 1.25, 1.5]
# noc_bws = [1.0, 2.0]

# hbm_bws = [i for i in range(100, 200+1, 10)]
# hbm_bws = hbm_bws[2:6] # TODO remove
models = ["llama2-13"]
# models = ["opt-30"]*3





all_data = {model: {} for model in models}
for m, model in enumerate(models):
    program_pickle_file= f"{out_location}outputs_icbm_{seq_length}_training/{num_cores}cores/{model}-b{batch_size}/program.pickle"
    prog = None
    with open(program_pickle_file, 'rb') as f:
        prog : DNNProgram.DNNProgram = pickle.load(f)
    total_flops = 0
    for op in prog.ops:
        flops = 2
        for dim_length in op.expr.dim_lengths:
            if dim_length != 0:
                flops *= dim_length
        total_flops += flops
    total_flops /= 1000000000
    for hbm_bw in hbm_bws:
        all_data[model][hbm_bw] = {}
        for noc_bw in noc_bws:
            all_data[model][hbm_bw][noc_bw] = {}
            all_data[model][hbm_bw][noc_bw]["Naive"] = []
            all_data[model][hbm_bw][noc_bw]["Min Cold"] = []
            all_data[model][hbm_bw][noc_bw]["Max Cold"] = []
            all_data[model][hbm_bw][noc_bw]["ICBM Ordered"] = []
            all_data[model][hbm_bw][noc_bw]["ICBM"] = []
            all_data[model][hbm_bw][noc_bw]["Ideal"] = []
            count=0
            for comp_bw in comp_bws:
                count+=1
                mesh_val = 0.0
                output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{hbm_bw}-{num_cores}-{kb}-{mesh_val}-{noc_bw}-{comp_bw}-training"
                with open(output_file, "r") as f:
                    for line in f:
                        if "naive_ms:" in line:
                            all_data[model][hbm_bw][noc_bw]["Naive"].append(total_flops / float(line.split(' ')[-1]))
                        elif "base_min_cold_ms:" in line:
                            all_data[model][hbm_bw][noc_bw]["Min Cold"].append(total_flops / float(line.split(' ')[-1]))
                        elif "base_max_cold_ms:" in line:
                            all_data[model][hbm_bw][noc_bw]["Max Cold"].append(total_flops / float(line.split(' ')[-1]))
                        elif "icbm_ordered_ms:" in line:
                            all_data[model][hbm_bw][noc_bw]["ICBM Ordered"].append(total_flops / float(line.split(' ')[-1]))
                        elif "icbm_ms:" in line:
                            all_data[model][hbm_bw][noc_bw]["ICBM"].append(total_flops / float(line.split(' ')[-1]))
                        elif "ideal_ms:" in line:
                            all_data[model][hbm_bw][noc_bw]["Ideal"].append(total_flops / float(line.split(' ')[-1]))

                    assert len(all_data[model][hbm_bw][noc_bw]["Naive"]) == count, f"Naive missing in {output_file}"
                    assert len(all_data[model][hbm_bw][noc_bw]["Min Cold"]) == count, f"Min Cold missing in {output_file}"
                    assert len(all_data[model][hbm_bw][noc_bw]["Max Cold"]) == count, f"Max Cold missing in {output_file}"
                    assert len(all_data[model][hbm_bw][noc_bw]["ICBM Ordered"]) == count, f"ICBM Ordered missing in {output_file}"
                    assert len(all_data[model][hbm_bw][noc_bw]["ICBM"]) == count, f"ICBM missing in {output_file}"
                    assert len(all_data[model][hbm_bw][noc_bw]["Ideal"]) == count, f"Ideal missing in {output_file}"
            all_data[model][hbm_bw][noc_bw]["Baseline"] = [max(a, b) for a, b in zip(all_data[model][hbm_bw][noc_bw]["Min Cold"], 
                                                                                     all_data[model][hbm_bw][noc_bw]["Max Cold"])]
            all_data[model][hbm_bw][noc_bw] = {impl[label]: all_data[model][hbm_bw][noc_bw][label] for label in impl}





mesh_data = {model: {} for model in models}
for m, model in enumerate(models):
    program_pickle_file= f"{out_location}outputs_icbm_{seq_length}_training/{num_cores}cores/{model}-b{batch_size}/program.pickle"
    prog = None
    with open(program_pickle_file, 'rb') as f:
        prog : DNNProgram.DNNProgram = pickle.load(f)
    total_flops = 0
    for op in prog.ops:
        flops = 2
        for dim_length in op.expr.dim_lengths:
            if dim_length != 0:
                flops *= dim_length
        total_flops += flops
    total_flops /= 1000000000
    for hbm_bw in hbm_bws:
        mesh_data[model][hbm_bw] = {}
        for noc_bw in noc_bws:
            mesh_data[model][hbm_bw][noc_bw] = {}
            mesh_data[model][hbm_bw][noc_bw]["Naive"] = []
            mesh_data[model][hbm_bw][noc_bw]["Min Cold"] = []
            mesh_data[model][hbm_bw][noc_bw]["Max Cold"] = []
            mesh_data[model][hbm_bw][noc_bw]["ICBM Ordered"] = []
            mesh_data[model][hbm_bw][noc_bw]["ICBM"] = []
            mesh_data[model][hbm_bw][noc_bw]["Ideal"] = []
            count=0
            for comp_bw in comp_bws:
                count+=1
                mesh_val = float(700/hbm_bw*noc_bw)
                mesh_val = mesh_val**0.5
                output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{hbm_bw}-{num_cores}-{kb}-{mesh_val}-{noc_bw}-{comp_bw}-training"
                with open(output_file, "r") as f:
                    for line in f:
                        if "naive_ms:" in line:
                            mesh_data[model][hbm_bw][noc_bw]["Naive"].append(total_flops / float(line.split(' ')[-1]))
                        elif "base_min_cold_ms:" in line:
                            mesh_data[model][hbm_bw][noc_bw]["Min Cold"].append(total_flops / float(line.split(' ')[-1]))
                        elif "base_max_cold_ms:" in line:
                            mesh_data[model][hbm_bw][noc_bw]["Max Cold"].append(total_flops / float(line.split(' ')[-1]))
                        elif "icbm_ordered_ms:" in line:
                            mesh_data[model][hbm_bw][noc_bw]["ICBM Ordered"].append(total_flops / float(line.split(' ')[-1]))
                        elif "icbm_ms:" in line:
                            mesh_data[model][hbm_bw][noc_bw]["ICBM"].append(total_flops / float(line.split(' ')[-1]))
                        elif "ideal_ms:" in line:
                            mesh_data[model][hbm_bw][noc_bw]["Ideal"].append(total_flops / float(line.split(' ')[-1]))
                    assert len(mesh_data[model][hbm_bw][noc_bw]["Naive"]) == count, f"Naive missing in {output_file}"
                    assert len(mesh_data[model][hbm_bw][noc_bw]["Min Cold"]) == count, f"Min Cold missing in {output_file}"
                    assert len(mesh_data[model][hbm_bw][noc_bw]["Max Cold"]) == count, f"Max Cold missing in {output_file}"
                    assert len(mesh_data[model][hbm_bw][noc_bw]["ICBM Ordered"]) == count, f"ICBM Ordered missing in {output_file}"
                    assert len(mesh_data[model][hbm_bw][noc_bw]["ICBM"]) == count, f"ICBM missing in {output_file}"
                    assert len(mesh_data[model][hbm_bw][noc_bw]["Ideal"]) == count, f"Ideal missing in {output_file}"
            mesh_data[model][hbm_bw][noc_bw]["Baseline"] = [max(a, b) for a, b in zip(mesh_data[model][hbm_bw][noc_bw]["Min Cold"],
                                                                                      mesh_data[model][hbm_bw][noc_bw]["Max Cold"])]
            mesh_data[model][hbm_bw][noc_bw] = {impl[label]: mesh_data[model][hbm_bw][noc_bw][label] for label in impl}



def plot_bw(model: str, hbm_bw: int, noc_bw: float, ax: plt.Axes, ylabel: bool = False, xlabel: bool = False, mesh=False):
    # Load all mesh_data

    if mesh:
        data = mesh_data
    else:
        data = all_data

        # do we still have any failing cases? idrk
        # if not count:
        #     exe_times['min_cold'].append(float('nan'))
        #     exe_times['max_cold'].append(float('nan'))
        #     exe_times['naive'].append(float('nan'))
        #     exe_times['icbm_ordered'].append(float('nan'))
        #     exe_times['icbm'].append(float('nan'))
        #     exe_times['ideal'].append(float('nan'))


    # print(len(hbm_bws))
    # print(len(exe_times['icbm']))
    # ax.plot(hbm_bws, exe_times['min_cold'], '.-', label='min_cold', color=colors[0])
    # print(len(noc_bws_tb))

    tf = [int(i*1000) for i in comp_bws]

    # ax.plot(comp_bws, data[model][hbm_bw][noc_bw]['Basic'], ':', label='Basic', color=colors[1], marker=markers[1], linewidth=2, markersize=8)
    ax.plot(tf, data[model][hbm_bw][noc_bw]['Static'], '--', label='Static', color=colors[0], marker=markers[0], linewidth=2, markersize=8)
    # ax.plot(comp_bws, data[model][hbm_bw][noc_bw]['ELK-Dyn'], '-.', label='ELK Ordered', color=colors[2], marker=markers[2], linewidth=2, markersize=8)
    ax.plot(tf, data[model][hbm_bw][noc_bw]['ELK-Full'], '-', label='ELK Full', color=colors[3], marker=markers[3], linewidth=2, markersize=8)
    ax.plot(tf, data[model][hbm_bw][noc_bw]['Ideal'], '--', label='Ideal', color=colors[4], marker=markers[4], zorder=0, linewidth=2, markersize=8)

    # ax.set_yscale("log")
    if xlabel:
        ax.set_xticklabels([])
        ax.set_xlabel(f"{int(noc_bw*32)}TB/s NoC", fontsize=25, labelpad=-143)
        

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)
    # ax.set_yticklabels(np.array(ax.get_yticks()).astype(int), position=(0.03, 0))
    # ax.set_xticklabels(np.array(ax.get_xticks()).astype(int), position=(0, 0.03))
    ax.set_ylim([300, 700])

    if ylabel:
        pass
    else:
        ax.set_yticklabels([])
        ax.set_ylabel(f"{int(hbm_bw)}GB/s\nHBM", fontsize=25, labelpad=-250)




if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='Draw Bandwith Figures')
    parser.add_argument("--num_cores", type=str, default="5888")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--kb", type=int, default=624)

    args = parser.parse_args()

    batch_size = args.batch_size
    seq_length = args.seq_length
    kb = args.kb
    num_cores = args.num_cores

    plt.rc('xtick', labelsize=23)
    plt.rc('ytick', labelsize=23)

    def draw_one(mesh):
        fig, ax = plt.subplots(2, len(hbm_bws), figsize=(7.3, 4.8))
        plt.subplots_adjust(top=0.83, bottom=0.16, left=0.15, right=0.88)
        plt.subplots_adjust(wspace=0.1, hspace=0.15)
        fig.text(0.525, 0.01, 'Available TFLOPS for MatMul', ha='center', fontsize=27)
        fig.text(0.03, 0.18, 'Achieved TFLOPS', ha='center', fontsize=26, rotation=90)
        
        for i, hbm_bw in enumerate(hbm_bws):
            for j, noc_bw in enumerate(noc_bws):
                ylabel = (j == 0)
                xlabel = (i == 0)
                plot_bw(model=models[0], hbm_bw=hbm_bw, noc_bw=noc_bw, ax=ax[i][j], ylabel=ylabel, xlabel=xlabel, mesh=mesh)

        ax[0][-1].legend(loc="upper right", fontsize=25, ncol=5, frameon=False,
                    handlelength=1.2, handletextpad=0.35, borderaxespad=0, labelspacing=0.1, columnspacing=1.2,
                    bbox_to_anchor=(1.13,1.6))

        # plt.legend(loc="upper right")
        description = f"{num_cores}cores-{kb}kb-b{batch_size}-{seq_length}seq"
        # plt.title(description)
        if mesh:
            plt.savefig(f"fig24-train-mesh-{description}.png")
            plt.savefig(f"fig24-train-mesh-{description}.pdf")
        else:
            plt.savefig(f"fig24-train-all-{description}.png")
            plt.savefig(f"fig24-train-all-{description}.pdf")
        # plt.savefig("pretty.png", pad_inches=0)

    draw_one(mesh=False)
    draw_one(mesh=True)
