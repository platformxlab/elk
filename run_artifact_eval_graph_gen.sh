#!/bin/bash

# fig. 17
echo "fig. 17"
python3.10 benchmark_scripts/draw_end2end.py --num_cores 5888 --kb 624 --bandwidth 16000

# fig. 18
echo "fig. 18"
python3.10 benchmark_scripts/draw_parts.py

# fig. 19
echo "fig. 19"
python3.10 benchmark_scripts/draw_bw_figures.py --num_cores 5888 --batch_size 32 --seq_length 2048 --kb 624

# fig. 20
echo "fig. 20"
python3.10 benchmark_scripts/draw_exec_breakdown_groupings.py --model llama2-13 --num_cores 5888 --batch_size 32 --seq_length 2048 --kb 624

# fig. 21
echo "fig. 21"
python3.10 benchmark_scripts/draw_noc_utilization.py --num_cores 5888 --batch_size 32 --seq_length 2048 --kb 624

# fig. 22
echo "fig. 22"
python3.10 benchmark_scripts/draw_noc_bw.py  --num_cores 5888 --batch_size 32 --seq_length 2048 --kb 624

# fig. 23 
echo "fig. 23"
python3.10 benchmark_scripts/draw_core_lines.py --batch_size 32 --seq_length 2048 --kb 624

# fig. 24
echo "fig. 24"
python3.10 benchmark_scripts/draw_training.py --num_cores 5888 --batch_size 2 --seq_length 2048 --kb 624
