# Elk: Exploring the Efficiency of Inter-core Connected AI Chips with Deep Learning Compiler Techniques (Artifact)

## 1. Abstract
This is the artifact of the [Elk paper](https://arxiv.org/abs/2507.11506), which is accepted at [MICRO'2025](https://microarch.org/micro58/).
In this artifact, we provide the source code of Elk's compilation, simulation, and evaluation framework. Then, we guide readers to explore how Elk improves the model serving performance on a variety of ICCA chips (i.e., Figure 17–24 in [our paper](https://arxiv.org/abs/2507.11506)). To run this artifact, please use a Linux machine with at least 200 GB of main memory and at least 20 GB of disk space.

## 2. Artifact Checklist
- **Algorithm**: Inductive tensor operator scheduling, cost-aware on-chip memory allocation, and ICCA chip design space exploration.
- **Neural Network Models**: Llama2-13B, Gemma2-27B, OPT-30B, Llama2-70B, and DIT-XL. Their execution graphs are included in the repo.
- **Run-time environment**: Ubuntu 20.04 or newer, Python 3.10.
- **Metrics**: Execution time, hardware utilization.
- **Output**: Trace files and result figures.
- **Experiments**: Generate experiments using supplied scripts.
- **How much main memory required (approximately)**: 200 GB
- **How much disk space required (approximately)**: 20 GB
- **How much time to prepare workflow (approximately)**: 10 minutes
- **How much time to complete experiments (approximately)**: 30 hours on a machine with 64 CPU cores and 200 GB main memory.

## 3. Description

### 3.1 Hardware Dependencies
The Elk simulation and evaluation framework can run on any x86 machine with at least 200 GB of main memory and at least 20 GB of disk space.

### 3.2 Software Dependencies
The framework needs a Linux environment (preferably Ubuntu) with Python 3.10 installed.

## 4. Installation
1. Start by downloading the Elk artifact from GitHub:
   ```bash
   git clone https://github.com/platformxlab/elk.git
   ```

2. Please make sure all prerequisites are successfully installed:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip tmux
   cd elk
   pip3 install -r requirements.txt
   ```

## 5. Experiment Workflow
To compile DL models into programs and obtain program execution traces from the Elk simulator, we provide a one-click script `benchmark_scripts/generate_data_from_sim.py` for you to launch all test cases in one place. However, the script may take more than 30 hours to finish, so we recommend using `tmux`:
```bash
tmux
```
Then, run the one-click script within the new tmux terminal:
```bash
python3 benchmark_scripts/generate_data_from_sim.py
```
To return from the pop-up terminal without pausing the script, press `Ctrl+B` and then press `D` on your keyboard. To attach back to the original tmux terminal where the script is running, use:
```bash
tmux attach -t 0
```
For more tips on using tmux, refer to [https://tmuxcheatsheet.com](https://tmuxcheatsheet.com).

### 5.1 Handle Errors
If the script encounters an error, the most common cause is that the artifact runs on too many CPU cores and overflows the main memory. In such events, go to `icbm_launch.py`, change the `CORE_REDUCE` macro in line 22 to a larger value (e.g., 8). Finally, attach back to the original tmux terminal and run again:
```bash
tmux attach -t 0
python3 benchmark_scripts/generate_data_from_sim.py
```
The script should automatically skip any completed test cases and resume from the failed one.

## 6. Evaluation and Expected Results
After the completion of all experiments, please run the following script to evaluate the results:
```bash
./run_artifact_eval_graph_gen.sh
```
This script gathers all data from the execution trace and draws all figures. To verify the results, one can compare the generated figures with those in the paper.
