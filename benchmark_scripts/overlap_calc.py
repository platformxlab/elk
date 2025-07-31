#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
out_location = ""

import pickle
from DNNProgram import DNNProgram

def IPU_Mk2_cycle_to_ms(cycles):
    return cycles / 1.325e6

logfile = "dump.log"

DEBUG = False
EPSILON = 0.0001

def get_breakdown_stats_percentage(logfile: str,
                                   model: str,
                                   bs: int,
                                   num_cores: int,
                                   seq_length: int,
                                   hbm_GBps: int,
                                   mod = False):

    program_pickle_file= f"{out_location}outputs_icbm_{seq_length}/{num_cores}cores/{model}-b{bs}/program.pickle"

    stats = {}

    f = open(program_pickle_file, 'rb')
    prog : DNNProgram = pickle.load(f)
    f.close()

    f = open(logfile, "r")

    while True:
        label = f.readline().strip(":\n")
        if not label: # basically EOF
            break

        exe_list_flat = f.readline().strip('[]()\n').split('), (')
        exe_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in exe_list_flat]

        load_list_flat = f.readline().strip('[]()\n').split('), (')
        load_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in load_list_flat]

        comp_shift_list_flat = f.readline().strip('[]()\n').split('), (')
        comp_shift_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in comp_shift_list_flat]
        # comp_shift_list = [(x / (x+y), y / (x+y)) if (x != 0 and y != 0) else (0, 0) for x, y in comp_shift_list] # make into %s
        # comp_shift_list = [(x / (x+y), y / (x+y)) if (x != 0 and y != 0) else (0, 0) for x, y in comp_shift_list] # make into %s
        # TODO -> how to manage this, still a bit unsure
        comp_shift_list = [((c + s) / (end-start), 1 - ((c + s) / (end-start))) for (c, s), (start, end) in zip(comp_shift_list, exe_list)]
        # comp_shift_list = [(c+s) / () for (c, s), (start, end) in zip(comp_shift_list, exe_list) ]

        # for cs, noc in comp_shift_list:
        #     print(cs, noc)
        #     assert abs(cs + noc - 1) < EPSILON
        # exit(0)

        f.readline() # idek what this is anymore

        order = []
        if label == "ICBM":
            order = [int(i) for i in f.readline().strip('[]\n').split(', ')]


        hbm_noc_list = []
        for i in range(len(prog.ops)):
            idx = order[i] if label == "ICBM" else i
            load_size_byte = prog.ops[idx].min_cold_size_bytes_per_core * prog.tot_num_cores
            load_time_ms = float(load_size_byte) / 1024./ 1024. / hbm_GBps
            load_time_cycle = load_time_ms * 1.325e6
            hbm_noc_list.append((load_time_cycle, (load_list[i][1] - load_list[i][0]) - load_time_cycle))
        hbm_noc_list = [(x / (x+y), y / (x+y)) if (x != 0 and y != 0) else (1, 0) for x, y in hbm_noc_list] # make into %s

        for hbm, noc in hbm_noc_list:
            if not abs(hbm + noc - 1) < EPSILON:
                print(hbm, noc)
            assert abs(hbm + noc - 1) < EPSILON

        for cs, noc in comp_shift_list:
            assert cs + noc == 1

        load_ptr = 0
        exe_ptr = 0

        load_ts = load_list[0][0]
        exe_ts = exe_list[0][0]

        load_only = 0
        exe_only = 0
        overlapped = 0

        intervals = []
        # since one of load_ptr, exe_ptr is incremented every time, I think each iter can count as an interval
        while (exe_ptr < len(exe_list)) and (load_ptr < len(load_list)):
            interval_data = {}

            if load_ts == exe_ts:
                if DEBUG:
                    print("BOTH")
                end_ts = min(load_list[load_ptr][1], exe_list[exe_ptr][1])
                overlapped += (end_ts - exe_ts) # time until one finishes is overlapped
                # print("adding to overlapped: {}")

                interval_data["duration"] = end_ts - load_ts
                interval_data["compute"] = comp_shift_list[exe_ptr][0]
                interval_data["shift"] = comp_shift_list[exe_ptr][1]
                interval_data["hbm"] = hbm_noc_list[load_ptr][0]
                interval_data["load_noc"] = hbm_noc_list[load_ptr][1]

                load_ts, exe_ts = end_ts, end_ts # now both are at end_ts
                if end_ts == load_list[load_ptr][1]: # if end_ts coencides w/ end of load, skip to next
                    load_ptr += 1
                    if load_ptr == len(load_list):
                        break
                    load_ts = load_list[load_ptr][0]
                if end_ts == exe_list[exe_ptr][1]: # if end_ts coencides w/ end of exec, skip to next
                    exe_ptr += 1
                    if exe_ptr == len(exe_list):
                        break
                    exe_ts = exe_list[exe_ptr][0]
            elif load_ts < exe_ts:
                if DEBUG:
                    print("LOAD ONLY")

                interval_data["compute"] = 0.0
                interval_data["shift"] = 0.0
                interval_data["hbm"] = hbm_noc_list[load_ptr][0]
                interval_data["load_noc"] = hbm_noc_list[load_ptr][1]

                if load_list[load_ptr][1] <= exe_ts: # if load ends before more exe happens
                    interval_data["duration"] = (load_list[load_ptr][1] - load_ts)
                    load_only += (load_list[load_ptr][1] - load_ts)
                    load_ptr += 1
                    if load_ptr == len(load_list):
                        break
                    load_ts = load_list[load_ptr][0]
                else:
                    interval_data["duration"] = (exe_ts - load_ts)

                    load_only += (exe_ts - load_ts)
                    load_ts = exe_ts
            elif exe_ts < load_ts:
                if DEBUG:
                    print("EXE ONLY")

                interval_data["compute"] = comp_shift_list[exe_ptr][0]
                interval_data["shift"] = comp_shift_list[exe_ptr][1]
                interval_data["hbm"] = 0.0
                interval_data["load_noc"] = 0.0

                if exe_list[exe_ptr][1] <= load_ts: # if exe ends before next load begins
                    interval_data["duration"] = (exe_list[exe_ptr][1] - exe_ts)
                    exe_only += (exe_list[exe_ptr][1] - exe_ts)
                    exe_ptr += 1
                    if exe_ptr == len(exe_list):
                        break
                    exe_ts = exe_list[exe_ptr][0]
                else:
                    interval_data["duration"] = (load_ts - exe_ts)
                    exe_only += (load_ts - exe_ts)
                    exe_ts = load_ts
            if DEBUG:
                print(f"{load_ptr}: ({load_list[load_ptr][0]}, {load_list[load_ptr][1]}) -> {load_ts}")
                print(f"{exe_ptr}: ({exe_list[exe_ptr][0]}, {exe_list[exe_ptr][1]}) -> {exe_ts}")
                val = input("Press any key to continue:")

            intervals.append(interval_data)

        # # TODO -> interval data stuff here
        # if exe_ptr < len(exe_list): # handle leftover exe
        #     exe_only += (exe_list[exe_ptr][1] - exe_ts)
        #     exe_only += sum([end - start for (start, end) in exe_list[exe_ptr+1:]])

        # for interval in exe_list[exe_ptr:]:
        while exe_ptr < len(exe_list): # handle leftover exe
            interval_data["compute"] = comp_shift_list[exe_ptr][0]
            interval_data["shift"] = comp_shift_list[exe_ptr][1]
            interval_data["hbm"] = 0.0
            interval_data["load_noc"] = 0.0
            interval_data["duration"] = (exe_list[exe_ptr][1] - exe_ts)

            exe_only += (exe_list[exe_ptr][1] - exe_ts)
            exe_ptr += 1
            if exe_ptr != len(exe_list):
                exe_ts = exe_list[exe_ptr][0]

        while load_ptr < len(load_list):
            interval_data["compute"] = 0.0
            interval_data["shift"] = 0.0
            interval_data["hbm"] = hbm_noc_list[load_ptr][0]
            interval_data["load_noc"] = hbm_noc_list[load_ptr][1]
            interval_data["duration"] = (load_list[load_ptr] - load_ts)

            load_only += (load_list[load_ptr][1] - load_ts) # TODO don't need this anymore
            load_ptr += 1
            if load_ptr != len(load_list):
                load_ts = load_list[load_ptr][0]

        # TODO repeat the above for loads
        # if load_ptr < len(load_list): # handle leftover loads
        #     load_only += (load_list[load_ptr][1] - load_ts)
        #     load_only += sum([end - start for (start, end) in load_list[load_ptr+1:]])

        cycles = -load_list[0][0]

        # print("all hbm", sum([x[0] for x in hbm_noc_list]))
        # print("all load noc", sum([x[1] for x in hbm_noc_list]))
        # print("all compute/shift", sum(x[0] for x in comp_shift_list))
        # print("all compute noc", sum(x[1] for x in comp_shift_list))
        overlapped = sum([data["hbm"] * data["compute"] * data["duration"] for data in intervals])
        load_only = sum([(data["hbm"] - (data["hbm"] * data["compute"])) * data["duration"] for data in intervals])
        exe_only = sum([(data["compute"] - (data["hbm"] * data["compute"])) * data["duration"] for data in intervals])
        noc_only = cycles - (overlapped + load_only + exe_only)
        exe_only, overlapped, noc_only = overlap(label, load_only, exe_only, overlapped, noc_only, mod)

        # print(overlapped, load_only, exe_only, noc_only)

        # exe_cycles = sum([exe[1]-exe[0] for exe in exe_list])
        # load_cycles = sum([load[1]-load[0] for load in load_list])
        # cycles = -load_list[0][0]

        # assert abs((load_only + overlapped) - load_cycles) < EPSILON
        # assert abs((exe_only + overlapped) - exe_cycles) < EPSILON
        # assert abs((load_only + exe_only + overlapped) -cycles) < EPSILON

        load_ms = IPU_Mk2_cycle_to_ms(load_only)
        exe_ms = IPU_Mk2_cycle_to_ms(exe_only)
        overlapped_ms = IPU_Mk2_cycle_to_ms(overlapped)
        noc_ms = IPU_Mk2_cycle_to_ms(noc_only)

        stats[label] = [(load_ms, exe_ms, overlapped_ms, noc_ms)]

    return stats

def get_breakdown_stats_simple(logfile: str,
                        model: str,
                        bs: int,
                        num_cores: int,
                        seq_length: int,
                        hbm_GBps: int):

    program_pickle_file= f"/dev/shm/outputs_icbm_{seq_length}/{num_cores}cores/{model}-b{bs}/program.pickle"

    stats = {}

    fp = open(program_pickle_file, 'rb')
    prog : DNNProgram = pickle.load(fp)
    fp.close()
    
    f = open(logfile, "r")

    cur_pos = 0 # need to track this b/c of inconsistencies w/ format
    while True:
        label = f.readline().strip(":\n")
        if not label: # basically EOF
            break

        f.readline()

        load_list_flat = f.readline().strip('[]()\n').split('), (')
        load_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in load_list_flat]

        comp_shift_list_flat = f.readline().strip('[]()\n').split('), (')
        comp_shift_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in comp_shift_list_flat]

        f.readline() # Get rid of IPU cmp shift data TODO -> do this automatically

        order = None
        if label == "ICBM":
            # f.readline()
            # line = f.readline().strip('[]\n').split(', ')
            # print(line)
            # exit(0)
            order = [int(i) for i in f.readline().strip('[]\n').split(', ')]

        hbm_list = []
        for i in range(len(prog.ops)):
            load_size_byte = prog.ops[i].min_cold_size_bytes_per_core * prog.tot_num_cores
            load_time_ms = float(load_size_byte) / 1024./ 1024. / hbm_GBps
            load_time_cycle = load_time_ms * 1.325e6

            hbm_list.append(load_time_cycle)

        exe_list = [c + s for c, s in comp_shift_list]

        load_ms = IPU_Mk2_cycle_to_ms(sum(hbm_list))
        exe_ms = IPU_Mk2_cycle_to_ms(sum(exe_list))
        total_ms = IPU_Mk2_cycle_to_ms(-load_list[0][0])
        overlapped_ms = load_ms + exe_ms - total_ms

        stats[label] = [(load_ms-overlapped_ms, exe_ms-overlapped_ms, overlapped_ms)]
    return stats

def overlap(mod, in1, inout1, inout2, inout3, is_overlap, 
            double_buffer_space_ratio=0.9):
    if is_overlap:
        if mod == "Naive":
            swap = in1**double_buffer_space_ratio
            inout1 -= swap
            inout2 += swap
        if mod != "ICBM":
            swap = 2*inout3*double_buffer_space_ratio
            if mod == "Naive": inout1 -= swap
            else: inout2 -= swap
            inout3 += swap
    return inout1, inout2, inout3

def get_breakdown_stats(logfile: str):
    stats = {}

    f = open(logfile, "r")

    cur_pos = 0 # need to track this b/c of inconsistencies w/ format
    while True:
        label = f.readline().strip(":\n")
        if not label: # basically EOF
            break
        exe_list_flat = f.readline().strip('[]()\n').split('), (')
        exe_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in exe_list_flat]

        load_list_flat = f.readline().strip('[]()\n').split('), (')
        load_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in load_list_flat]

        # if label == "Min Cold":
        f.readline() # Get rid of IPU cmp shift data TODO -> do this automatically
        f.readline() # Get rid of IPU cmp shift data TODO -> do this automatically

        if label == "ICBM":
            f.readline() # order

        # time = min(exe_list[0][0], load_list[0][0])

        load_ptr = 0
        exe_ptr = 0

        load_ts = load_list[0][0]
        exe_ts = exe_list[0][0]

        load_only = 0
        exe_only = 0
        overlapped = 0

        while (exe_ptr < len(exe_list)) and (load_ptr < len(load_list)):
            if DEBUG:
                print(f"{load_ptr}: ({load_list[load_ptr][0]}, {load_list[load_ptr][1]}) -> {load_ts}")
                print(f"{exe_ptr}: ({exe_list[exe_ptr][0]}, {exe_list[exe_ptr][1]}) -> {exe_ts}")
            if load_ts == exe_ts:
                if DEBUG:
                    print("BOTH")
                end_ts = min(load_list[load_ptr][1], exe_list[exe_ptr][1])
                overlapped += (end_ts - exe_ts) # time until one finishes is overlapped
                # print("adding to overlapped: {}")
                load_ts, exe_ts = end_ts, end_ts # now both are at end_ts
                if end_ts == load_list[load_ptr][1]: # if end_ts coencides w/ end of load, skip to next
                    load_ptr += 1
                    if load_ptr == len(load_list):
                        break
                    load_ts = load_list[load_ptr][0]
                if end_ts == exe_list[exe_ptr][1]: # if end_ts coencides w/ end of exec, skip to next
                    exe_ptr += 1
                    if exe_ptr == len(exe_list):
                        break
                    exe_ts = exe_list[exe_ptr][0]
            elif load_ts < exe_ts:
                if DEBUG:
                    print("LOAD ONLY")
                if load_list[load_ptr][1] <= exe_ts: # if load ends before more exe happens
                    load_only += (load_list[load_ptr][1] - load_ts)
                    load_ptr += 1
                    if load_ptr == len(load_list):
                        break
                    load_ts = load_list[load_ptr][0]
                else:
                    load_only += (exe_ts - load_ts)
                    load_ts = exe_ts
            elif exe_ts < load_ts:
                if DEBUG:
                    print("EXE ONLY")
                if exe_list[exe_ptr][1] <= load_ts: # if exe ends before next load begins
                    exe_only += (exe_list[exe_ptr][1] - exe_ts)
                    exe_ptr += 1
                    if exe_ptr == len(exe_list):
                        break
                    exe_ts = exe_list[exe_ptr][0]
                else:
                    exe_only += (load_ts - exe_ts)
                    exe_ts = load_ts
            if DEBUG:
                print(f"{load_ptr}: ({load_list[load_ptr][0]}, {load_list[load_ptr][1]}) -> {load_ts}")
                print(f"{exe_ptr}: ({exe_list[exe_ptr][0]}, {exe_list[exe_ptr][1]}) -> {exe_ts}")
                val = input("Press any key to continue:")


        # TODO I'm pretty sure this is wrong since it doesn't go to 0 as ts always
        if exe_ptr < len(exe_list): # handle leftover exe
            exe_only += (exe_list[exe_ptr][1] - exe_ts)
            exe_only += sum([end - start for (start, end) in exe_list[exe_ptr+1:]])

        if load_ptr < len(load_list): # handle leftover loads
            load_only += (load_list[load_ptr][1] - load_ts)
            load_only += sum([end - start for (start, end) in load_list[load_ptr+1:]])

        exe_cycles = sum([exe[1]-exe[0] for exe in exe_list])
        load_cycles = sum([load[1]-load[0] for load in load_list])
        cycles = -load_list[0][0]

        # assert abs((load_only + overlapped) - load_cycles) < EPSILON
        # assert abs((exe_only + overlapped) - exe_cycles) < EPSILON
        # assert abs((load_only + exe_only + overlapped) -cycles) < EPSILON

        load_ms = IPU_Mk2_cycle_to_ms(load_only)
        exe_ms = IPU_Mk2_cycle_to_ms(exe_only)
        overlapped_ms = IPU_Mk2_cycle_to_ms(overlapped)

        stats[label] = [(load_ms, exe_ms, overlapped_ms)]
        # print(label, load_only, exe_only, overlapped)
    return stats
