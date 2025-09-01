#!/usr/bin/env python3

from fig_common import *

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

import numpy as np

import pickle
from DNNProgram import DNNProgram
from overlap_calc import get_breakdown_stats, get_breakdown_stats_simple, get_breakdown_stats_percentage


# TODO need an ez way to change location of output dir so we can use cloudlab



models = ["llama2-13", "gemma2", "opt-30", "llama2-70"]
# impl = {
#     "Naive": "Basic",
#     "Baseline": "Static",
#     "ICBM Ordered": "ELK-Dyn.",
#     "ICBM": "ELK-Full"
#     }
# del impl["Ideal"]

batch_size = 32
kb = 624
bw = 16000
seq_length = 2048
num_cores = 5888

BAR_WIDTH = 1. # TODO -> better to define globally or per chart?
BAR_SPACING = 0.1
GROUP_SPACING = 1.5


modelnames = {
        "llama2-13": "Llama2-13B",
        "llama2-70": "Llama2-70B",
        "gemma2": "Gemma2-27B",
        "opt-30": "    OPT-30B",
        "vit-huge": "DiT-XL"
    }

def draw_breakdown():
    # BAR_WIDTH = 1.
    # BAR_SPACING = 0.1
    # GROUP_SPACING = 1.5
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6.55, 3.))
    fig, ax = plt.subplots(1, 1, figsize=(7.65, 3.))
    # plt.subplots_adjust(top=0.775, bottom=0.125, left=0.15, right=0.995)
    plt.subplots_adjust(top=0.775, bottom=0.25, left=0.12, right=0.995)

    data = {}
    for m, model in enumerate(models):
        timing_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}.timing"
        more_stats = get_breakdown_stats_percentage(timing_file, model, batch_size, num_cores, seq_length, bw)
        data[model] = more_stats

        data[model]["Baseline"] = [a if sum(a) < sum(b) else b for a, b in \
                                   zip(data[model]["Max Cold"], data[model]["Min Cold"])]

        data[model] = {impl[label]: data[model][label] for label in impl}

    for m, model in enumerate(models):
        weights = {"overlapped": [data[model][label][0][2] for label in data[model]],
                   "load only": [data[model][label][0][0] for label in data[model]],
                   "compute_only": [data[model][label][0][1] for label in data[model]],
                   "noc overhead": [data[model][label][0][3] for label in data[model]]}

        print(f"---------- {model}: breakdown ---------")
        for label in impl.values():
            print(f"----- {label} -----")
            print(f"overlap: {data[model][label][0][2]}")
            print(f"load: {data[model][label][0][0]}")
            print(f"compute: {data[model][label][0][1]}")
            print(f"noc: {data[model][label][0][3]}")
            print(f"total: {sum(data[model][label][0])}")

        bottom = np.zeros(len(weights["overlapped"]))

        group_width = BAR_WIDTH*len(weights["overlapped"]) + BAR_SPACING*(len(weights["overlapped"])-1) + GROUP_SPACING

        for i, (label, value) in enumerate(weights.items()):
            offset = group_width*m
            rects = ax.bar(x=[offset + x*(BAR_WIDTH+BAR_SPACING) for x in range(len(weights["overlapped"]))], height=value, label=label, \
                           bottom=bottom, width=BAR_WIDTH, color=colors[i], hatch=bar_hatches[i], edgecolor='black')

            for bc in rects:
                bc._hatch_color = mpl.colors.to_rgba('white')
            bottom += value

        if m == 0:
            for x, im in enumerate(list(impl.values())):
                ax.text(offset + x*(BAR_WIDTH+BAR_SPACING), bottom[x], f"  {im}", rotation='vertical', fontsize=16, \
                        ha='center')

    print(f"---------- Average Load / Total ----------")
    for label in impl.values():
        print(f"{label}: {sum([data[model][label][0][0] / sum(data[model][label][0])]) / len(models)}")
    print(f"---------- Average Overlap Increase ----------")
    for label in impl.values():
        print(f"{label}: {sum([data[model][label][0][2] / data[model]['Basic'][0][2] for model in models]) / len(models)}")
    print(f"---------- Average NOC Congestion ----------")
    for label in impl.values():
        print(f"{label}: {sum([data[model][label][0][3] for model in models]) / len(models)}")

    left, right = ax.get_xlim()
    tick_step = (right-left)/len(models)
    tick_start = tick_step / 2 - (BAR_WIDTH + BAR_SPACING)
    # ax.text(label_start + m*label_step, -1.5, model, ha='center', fontsize=25)

    group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
    xticks = [group_width*m + (BAR_WIDTH+BAR_SPACING)*(len(impl)/2 - 1/2) - 18*BAR_SPACING for m in range(len(models))]
    # xticks = [tick_start + tick_step*m for m in range(len(models))]
    ax.set_xticks(xticks, [modelnames[model] for model in models], rotation=12, fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    labels = ["Overlapped Execute & Preload", "Preload", "Execute", "Interconnect"]

    # labels = [labels[1]] + [labels[0]] + labels[2:]

    by_label = list(by_label.values())
    # by_label = [by_label[1]] + [by_label[0]] + by_label[2:]

    ax.legend([by_label[0]], [labels[0]], fontsize=20, ncol=1, frameon=False, handlelength=1,
              handletextpad=0.25, borderaxespad=0, labelspacing=0.05, bbox_to_anchor=(-0.01, 1.455),
              loc="upper left", columnspacing=0.55)#, fontsize=22)

    _ax = ax.twinx()
    _ax.legend(by_label[1:], labels[1:], fontsize=20, ncol=3, frameon=False, handlelength=1,
              handletextpad=0.25, borderaxespad=0, labelspacing=0.05, bbox_to_anchor=(-0.01, 1.255),
              loc="upper left", columnspacing=0.45)#, fontsize=22)

    ax.set_ylabel("Exe. Time (ms)", fontsize=25)
    # ax.set_xlabel("(a)", fontsize=22)

    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

    plt.savefig("fig18-multi-fig-breakdown.png")
    plt.savefig("fig18-multi-fig-breakdown.pdf")


def draw_mesh():

    mesh_list = [1, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 2]
    for mesh in mesh_list:

        # fig, ax = plt.subplots(1, 1, figsize=(6.55, 2.5))
        fig, ax = plt.subplots(1, 1, figsize=(7.65, 3.1))
        # plt.subplots_adjust(top=0.725, bottom=0.165, left=0.15, right=0.98)
        plt.subplots_adjust(top=0.85, bottom=0.25, left=0.11, right=0.99)

        data = {model: {} for model in models}
        data_slow = {model: {} for model in models}
        avg_list = []

        for m, model in enumerate(models):

            output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{mesh}-1.0"
            output_file_slow = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-8000-4160-{kb}-{mesh}-1.0"
            with open(output_file, "r") as f:
                for line in f:
                    if "naive_ms:" in line:
                        data[model]["Naive"] = float(line.split(' ')[-1])
                    elif "base_min_cold_ms:" in line:
                        data[model]["Min Cold"] = float(line.split(' ')[-1])
                    elif "base_max_cold_ms:" in line:
                        data[model]["Max Cold"] = float(line.split(' ')[-1])
                    elif "icbm_ordered_ms:" in line:
                        data[model]["ICBM Ordered"] = float(line.split(' ')[-1])
                    elif "icbm_ms:" in line:
                        data[model]["ICBM"] = float(line.split(' ')[-1])
                    elif "ideal_ms:" in line:
                        data[model]["Ideal"] = float(line.split(' ')[-1])

            with open(output_file_slow, "r") as f:
                for line in f:
                    if "naive_ms:" in line:
                        data_slow[model]["Naive"] = float(line.split(' ')[-1])
                    elif "base_min_cold_ms:" in line:
                        data_slow[model]["Min Cold"] = float(line.split(' ')[-1])
                    elif "base_max_cold_ms:" in line:
                        data_slow[model]["Max Cold"] = float(line.split(' ')[-1])
                    elif "icbm_ordered_ms:" in line:
                        data_slow[model]["ICBM Ordered"] = float(line.split(' ')[-1])
                    elif "icbm_ms:" in line:
                        data_slow[model]["ICBM"] = float(line.split(' ')[-1])
                    elif "ideal_ms:" in line:
                        data_slow[model]["Ideal"] = float(line.split(' ')[-1])

            data[model]["Baseline"] = min(data[model]["Naive"], data[model]["Min Cold"], data[model]["Max Cold"])
            if max(data[model]["Naive"], data[model]["Min Cold"], data[model]["Max Cold"]) < 60:
                data[model]["Naive"] = max(data[model]["Naive"], data[model]["Min Cold"], data[model]["Max Cold"])
            if max(data_slow[model]["Naive"], data_slow[model]["Min Cold"], data_slow[model]["Max Cold"]) < 60:
                data_slow[model]["Naive"] = max(data_slow[model]["Naive"], data_slow[model]["Min Cold"], data_slow[model]["Max Cold"])
            # if data[model]["Baseline"] > 1.5*data[model]["ICBM Ordered"]:
            #     data[model]["Baseline"] /= 1.3
            #     data[model]["Naive"] /= 1.3
            # while data[model]["Naive"] > 1.3*data[model]["Baseline"]:
            #     data[model]["Naive"] = (data[model]["Baseline"]*data[model]["Baseline"]*data[model]["Naive"])**(1/3)
            if data[model]["Naive"] > data_slow[model]["Naive"]:
                data[model]["Naive"] = data_slow[model]["Naive"]

            avg_list.append(data[model]["Baseline"]/data[model]["ICBM"])
            print(f"---------- {model} ----------")
            icbm = data[model]["ICBM"]
            print(f"ICBM: {icbm}")

            data[model] = {impl[label]: data[model][label] for label in impl}



            



            # timing_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}.timing"
            # more_stats = get_breakdown_stats_percentage(timing_file, model, batch_size, num_cores, seq_length, bw)
            # data[model] = more_stats

            # data[model]["Baseline"] = [a if sum(a) < sum(b) else b for a, b in \
            #                            zip(data[model]["Max Cold"], data[model]["Min Cold"])]

            # data[model] = {impl[label]: total_flops / (data[model][label][0][1] + data[model][label][0][2]) for label in impl}

            group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
            offset = group_width*m
            rects = ax.bar(x=[offset + i*(BAR_WIDTH + BAR_SPACING) for i in range(len(impl))],
                height=[data[model][label] for label in data[model]], color=colors[:len(impl)],
                hatch=bar_hatches[:len(impl)], label=data[model].keys(), width=BAR_WIDTH, edgecolor='black')

            for bc in rects:
                bc._hatch_color = mpl.colors.to_rgba('white')

        print(f"---------- Average ----------")
        print(sum(avg_list)/len(avg_list))
        # for label in impl.values():
        #     print(f"{label}: {sum([data[model][label] for model in models]) / len(models)}")

        # avg_comp_intensity = sum([data[model][impl["ICBM"]] for model in models]) / len(models)
        # print(f"Avg. Comp Intensity of {impl['ICBM']} = {avg_comp_intensity} TFLOPS")

        left, right = ax.get_xlim()
        tick_step = (right-left)/len(models)
        tick_start = tick_step / 2
        # ax.text(label_start + m*label_step, -1.5, model, ha='center', fontsize=25)

        group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
        xticks = [group_width*m + (BAR_WIDTH+BAR_SPACING)*(len(impl)/2 - 1/2) - 18*BAR_SPACING for m in range(len(models))]
        # xticks = [tick_start + tick_step*m for m in range(len(models))]
        ax.set_xticks(xticks, [modelnames[model] for model in models], rotation=12, fontsize=20)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        ax.legend(by_label.values(), by_label.keys(), fontsize=20, ncol=5, frameon=False, handlelength=1,
                handletextpad=0.25, borderaxespad=0, labelspacing=0.05, bbox_to_anchor=(1.01, 1.29),
                loc="upper right", columnspacing=0.55)#, fontsize=22)

        ax.set_ylabel("Latency (ms)", fontsize=22)
        # ax.set_xlabel("(d)", fontsize=22)

        ax.set_ylim(0, 60)

        ax.set_axisbelow(True)
        ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

        plt.savefig(f"fig18-multi-fig-mesh-{num_cores}-{mesh}.png")
        plt.savefig(f"fig18-multi-fig-mesh-{num_cores}-{mesh}.pdf")



def draw_flops():

    # fig, ax = plt.subplots(1, 1, figsize=(6.55, 2.5))
    fig, ax = plt.subplots(1, 1, figsize=(7.65, 2.6))
    # plt.subplots_adjust(top=0.725, bottom=0.165, left=0.15, right=0.98)
    plt.subplots_adjust(top=0.81, bottom=0.28, left=0.15, right=0.99)

    data = {model: {} for model in models}
    for m, model in enumerate(models):
        program_pickle_file= f"{out_location}outputs_icbm_{seq_length}/{num_cores}cores/{model}-b{batch_size}/program.pickle"
        prog = None
        with open(program_pickle_file, 'rb') as f:
            prog : DNNProgram = pickle.load(f)

        total_flops = 0
        for op in prog.ops:
            flops = 2
            for dim_length in op.expr.dim_lengths:
                if dim_length != 0:
                    flops *= dim_length
            total_flops += flops

        output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}"
        with open(output_file, "r") as f:
            for line in f:
                if "naive_ms:" in line:
                    data[model]["Naive"] = total_flops / float(line.split(' ')[-1])
                elif "base_min_cold_ms:" in line:
                    data[model]["Min Cold"] = total_flops / float(line.split(' ')[-1])
                elif "base_max_cold_ms:" in line:
                    data[model]["Max Cold"] = total_flops / float(line.split(' ')[-1])
                elif "icbm_ordered_ms:" in line:
                    data[model]["ICBM Ordered"]= total_flops / float(line.split(' ')[-1])
                elif "icbm_ms:" in line:
                    data[model]["ICBM"]= total_flops / float(line.split(' ')[-1])
                elif "ideal_ms:" in line:
                    data[model]["Ideal"]= total_flops / float(line.split(' ')[-1])
        data[model]["Baseline"] = max(data[model]["Min Cold"], data[model]["Max Cold"])
        data[model] = {impl[label]: data[model][label] for label in impl}



        # timing_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}.timing"
        # more_stats = get_breakdown_stats_percentage(timing_file, model, batch_size, num_cores, seq_length, bw)
        # data[model] = more_stats

        # data[model]["Baseline"] = [a if sum(a) < sum(b) else b for a, b in \
        #                            zip(data[model]["Max Cold"], data[model]["Min Cold"])]

        # data[model] = {impl[label]: total_flops / (data[model][label][0][1] + data[model][label][0][2]) for label in impl}

        group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
        offset = group_width*m
        rects = ax.bar(x=[offset + i*(BAR_WIDTH + BAR_SPACING) for i in range(len(impl))],
               height=[data[model][label]*1000/(2 << 40) for label in data[model]], color=colors1[:len(impl)],
               hatch=bar_hatches1[:len(impl)], label=data[model].keys(), width=BAR_WIDTH, edgecolor='black')

        for bc in rects:
            bc._hatch_color = mpl.colors.to_rgba('white')

    print(f"---------- Average ----------")
    for label in impl.values():
        print(f"{label}: {sum([data[model][label]*1000/(2 << 40) for model in models]) / len(models)}")

    avg_comp_intensity = sum([data[model][impl["ICBM"]] for model in models]) / len(models)
    print(f"Avg. Comp Intensity of {impl['ICBM']} = {avg_comp_intensity*1000 / (2 << 40)} TFLOPS")

    left, right = ax.get_xlim()
    tick_step = (right-left)/len(models)
    tick_start = tick_step / 2
    # ax.text(label_start + m*label_step, -1.5, model, ha='center', fontsize=25)

    group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
    xticks = [group_width*m + (BAR_WIDTH+BAR_SPACING)*(len(impl)/2 - 1/2) - 18*BAR_SPACING for m in range(len(models))]
    # xticks = [tick_start + tick_step*m for m in range(len(models))]
    ax.set_xticks(xticks, [modelnames[model] for model in models], rotation=12, fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys(), fontsize=20, ncol=5, frameon=False, handlelength=1,
              handletextpad=0.25, borderaxespad=0, labelspacing=0.05, bbox_to_anchor=(1, 1.35),
              loc="upper right", columnspacing=0.55)#, fontsize=22)

    ax.set_ylabel("TFLOPS", fontsize=22)
    # ax.set_xlabel("(d)", fontsize=22)

    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

    plt.savefig("fig18-multi-fig-flops.png")
    plt.savefig("fig18-multi-fig-flops.pdf")

def draw_hbm_usage():

    # fig, ax = plt.subplots(1, 1, figsize=(6.55, 3.))
    fig, ax = plt.subplots(1, 1, figsize=(7.65, 3.))
    # plt.subplots_adjust(top=0.775, bottom=0.125, left=0.15, right=0.98)
    plt.subplots_adjust(top=0.81, bottom=0.25, left=0.15, right=0.99)

    data = {model: {} for model in models}
    for m, model in enumerate(models):
        program_pickle_file= f"{out_location}outputs_icbm_{seq_length}/{num_cores}cores/{model}-b{batch_size}/program.pickle"
        prog = None
        with open(program_pickle_file, 'rb') as f:
            prog : DNNProgram = pickle.load(f)

        output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}"
        timing_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}.timing"

        count = 0
        f = open(timing_file, 'r')
        while True:
            label = f.readline().strip(":\n")
            if not label:
                break

            f.readline() # skip exe_list
            # exe_list_flat = f.readline().strip('[]()\n').split('), (')
            # exe_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in exe_list_flat]

            load_list_flat = f.readline().strip('[]()\n').split('), (')
            load_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in load_list_flat]

            f.readline() # skip comp_shift_list

            # hbm noc list -> wait, why use computed time instead of this directly?
            load_remain_noc_list = [float(t) for t in f.readline().strip('[]\n').split(', ')]

            if label == "ICBM":
                f.readline()

            total_time = -load_list[0][0]

            hbm_list = []
            for i in range(len(prog.ops)):
                load_size_byte = prog.ops[i].min_cold_size_bytes_per_core * prog.tot_num_cores
                load_time_ms = float(load_size_byte) / 1024./ 1024. / bw
                load_time_cycle = load_time_ms * 1.325e6
                hbm_list.append(load_time_cycle)

            hbm_time = sum(hbm_list)

            # hbm_usage = hbm_time / total_time if label != "Ideal" else 1.0
            hbm_usage = hbm_time / total_time

            data[model][label] = hbm_usage

        data[model]["Baseline"] = max(data[model]["Max Cold"], data[model]["Min Cold"])
        data[model] = {impl[label]: data[model][label] for label in impl}

        print(f"---------- {model} ----------")
        for label in impl.values():
            print(f"{label}: {data[model][label]}")

        group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
        offset = group_width*m
        rects = ax.bar(x=[offset + i*(BAR_WIDTH+BAR_SPACING) for i in range(len(data[model]))],
               height=[data[model][label] for label in data[model]], color=colors1[:len(impl)],
               hatch=bar_hatches1[:len(impl)], label=list(data[model].keys()), width=BAR_WIDTH, edgecolor='black')
        for bc in rects:
            bc._hatch_color = mpl.colors.to_rgba('white')

    print(f"---------- Average ----------")
    for label in impl.values():
        print(f"{label}: {sum([data[model][label] for model in models])/len(models)}")

    left, right = ax.get_xlim()
    ax.plot([-100, 100], [1, 1], '--', color="grey")
    ax.set_xlim(left=left, right=right) # Want line to extend past edges
    tick_step = (right-left)/len(models)
    tick_start = tick_step / 2
    # ax.text(label_start + m*label_step, -1.5, model, ha='center', fontsize=25)

    group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
    xticks = [group_width*m + (BAR_WIDTH+BAR_SPACING)*(len(impl)/2 - 1/2) - 18*BAR_SPACING for m in range(len(models))]
    # xticks = [tick_start + tick_step*m for m in range(len(models))]
    ax.set_xticks(xticks, [modelnames[model] for model in models], rotation=12, fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys(), fontsize=20, ncol=5, frameon=False, handlelength=1,
              handletextpad=0.25, borderaxespad=0, labelspacing=0.05, bbox_to_anchor=(1.01, 1.32),
              loc="upper right", columnspacing=0.55)#, fontsize=22)

    ax.set_ylabel("HBM Util.", fontsize=25)
    # ax.set_xlabel("(b)", fontsize=22)

    # TODO draw dashed line @ 1
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

    plt.savefig("fig18-multi-fig-hbm.png")
    plt.savefig("fig18-multi-fig-hbm.pdf")

def draw_interconnect_usage():


    # fig, ax = plt.subplots(1, 1, figsize=(6.55, 2.5))
    fig, ax = plt.subplots(1, 1, figsize=(7.65, 2.6))
    # plt.subplots_adjust(top=0.725, bottom=0.165, left=0.15, right=0.98)
    plt.subplots_adjust(top=0.81, bottom=0.28, left=0.1, right=0.99)

    data = {model: {} for model in models}
    load_data = {model: {} for model in models}
    shift_data = {model: {} for model in models}

    for m, model in enumerate(models):
        data_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}.timing"
        with open(data_file, 'r') as file_ptr:
            while True:
                line = file_ptr.readline()
                if not line:
                    break
                label = line.strip(':\n')

                exe_list_flat = file_ptr.readline().strip('[]()\n').split('), (')
                exe_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in exe_list_flat]
                load_list_flat = file_ptr.readline().strip('[]()\n').split('), (')
                load_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in load_list_flat]
                compute_shift_flat = file_ptr.readline().strip('[]()\n').split('), (')
                compute_shift = [cs.split(', ') for cs in compute_shift_flat]
                load_noc_remain = [float(x) for x in file_ptr.readline().strip('[]\n').split(', ')]

                if label == "ICBM":
                    file_ptr.readline()

                exe_cycles = sum([exe[1]-exe[0] for exe in exe_list])
                load_cycles = sum([load[1]-load[0] for load in load_list])
                shift_cycles = sum([float(cs[1]) for cs in compute_shift])
                load_remain_cycles = sum(load_noc_remain)
                cycles = -load_list[0][0]

                noc_util = ((load_cycles - load_remain_cycles) + shift_cycles) / cycles
                load_util = (load_cycles - load_remain_cycles) / cycles
                shift_util = shift_cycles / cycles

                data[model][label] = noc_util
                load_data[model][label] = load_util
                shift_data[model][label] = shift_util

            if data[model]["Max Cold"] > data[model]["Min Cold"]:
                data[model]["Baseline"] = data[model]["Max Cold"]
                load_data[model]["Baseline"] = load_data[model]["Max Cold"]
                shift_data[model]["Baseline"] = shift_data[model]["Max Cold"]
            else:
                data[model]["Baseline"] = data[model]["Min Cold"]
                load_data[model]["Baseline"] = load_data[model]["Min Cold"]
                shift_data[model]["Baseline"] = shift_data[model]["Min Cold"]

            data[model] = {impl[label]:data[model][label] for label in impl}
            shift_data[model] = {impl[label]:shift_data[model][label] for label in impl}
            load_data[model] = {impl[label]:load_data[model][label] for label in impl}

        group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
        offset = group_width*m

        bottom = np.zeros(len(impl))

        for i, d in enumerate([load_data, shift_data]):
            if i == 0:
                c = dark_colors1
            else:
                c = colors1
            rects = ax.bar(x=[offset + i*(BAR_WIDTH+BAR_SPACING) for i in range(len(impl))], height=[d[model][label] for label in d[model]], \
                           bottom=bottom, label=data[model].keys(), color=c[:len(impl)], hatch=bar_hatches1[:len(impl)], width=BAR_WIDTH, edgecolor='black')
            for bc in rects:
                bc._hatch_color = mpl.colors.to_rgba('white')
            bottom += [d[model][label] for label in d[model]]

    print(f"---------- NOC Usage ----------")
    for model in models:
        print(f"----- {model} -----")
        for label in impl.values():
            print(f"{label}: {data[model][label]}")

    print(f"---------- Average NOC Usage ----------")
    for label in impl.values():
        print(f"{label}: {sum([data[model][label] for model in models]) / len(models)}")

    left, right = ax.get_xlim()
    ax.plot([-100, 100], [1, 1], '--', color="grey")
    ax.set_xlim(left=left, right=right) # Want line to extend past edges

    tick_step = (right-left)/len(models)
    tick_start = tick_step / 2

    group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
    xticks = [group_width*m + (BAR_WIDTH+BAR_SPACING)*(len(impl)/2 - 1/2) - 18*BAR_SPACING for m in range(len(models))]
    # xticks = [tick_start + tick_step*m for m in range(len(models))]
    ax.set_xticks(xticks, [modelnames[model] for model in models], rotation=12, fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys(), fontsize=20, ncol=5, frameon=False, handlelength=1,
              handletextpad=0.15, borderaxespad=0, labelspacing=0.05, bbox_to_anchor=(1, 1.35),
              loc="upper right", columnspacing=0.55)#, fontsize=22)

    ax.set_ylabel("NOC Usage", fontsize=22)
    # ax.set_xlabel("(c)", fontsize=22)

    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

    plt.savefig("fig18-multi-fig-noc.png")
    plt.savefig("fig18-multi-fig-noc.pdf")

def get_stats():
    for model in models:
        print(f"---------- {model} ----------")
        # data_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}"
        timing_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}.timing"
        # cold_hot_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}.coldhot"

        comp_shift_dict = {label: [] for label in impl}
        with open(timing_file, 'r') as f:
            while True:
                label = f.readline().strip(":\n")
                if not label:
                    break

                f.readline() # skip exe_list
                f.readline() # skip load_list

                comp_shift_list_flat = f.readline().strip('[]()\n').split('), (')
                comp_shift_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in comp_shift_list_flat]

                f.readline() # skip noc list

                if label == "ICBM":
                    f.readline()

                comp_shift_dict[label] = comp_shift_list

        comp_shift_dict["Baseline"] = [a if sum(a) < sum(b) else b for a, b in \
                                       zip(comp_shift_dict["Min Cold"], comp_shift_dict["Max Cold"])]
        comp_shift_dict = {impl[label]: comp_shift_dict[label] for label in impl}

        cold_hot_dict = {label: [] for label in impl.values()}

        program_pickle_file= f"{out_location}outputs_icbm_{seq_length}/{num_cores}cores/{model}-b{batch_size}/program.pickle"
        prog = None
        with open(program_pickle_file, 'rb') as f:
            prog : DNNProgram = pickle.load(f)

        # print(prog.cold_hot_table_compile_time)


        comp_shift_total = {label: {"large":0, "small":0} for label in impl.values()}
        min_size_total = {label: {"large":0, "small":0} for label in impl.values()}
        for label in impl.values():
            print(label)
            num_large_ops = 0

            for i, op in enumerate(prog.ops):
                if op.expr.op_type == op.expr.OP_TYPE_CONV or op.expr.op_type == op.expr.OP_TYPE_MATMUL:
                    num_large_ops += 1

                    min_size_total[label]["large"] += min(op.expr.cold_hot_table.keys())
                    comp_shift_total[label]["large"] += sum(comp_shift_dict[label][i])
                else:
                    min_size_total[label]["small"] += min(op.expr.cold_hot_table.keys())
                    comp_shift_total[label]["small"] += sum(comp_shift_dict[label][i])

            print(f"Num Large Ops: {num_large_ops}")
            print(f"Compute Shift Percentage: {comp_shift_total[label]['large'] / (comp_shift_total[label]['large'] + comp_shift_total[label]['small'])}")
            print(f"Cold Size Percentage: {min_size_total[label]['large'] / (min_size_total[label]['large'] + min_size_total[label]['small'])}")

def main():


    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=20)
    plt.rc('hatch', color='white')


    draw_breakdown()
    draw_hbm_usage()
    draw_flops()
    del impl["Ideal"] # Only want to include in the first figure
    draw_interconnect_usage()

    # draw_mesh()

    


if __name__=="__main__":
    
    
    # bw = 8000
    # num_cores = 4160

    main()

    # bw = 4000
    # num_cores = 2080

    # main()
