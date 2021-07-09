#!/bin/bash
#BSUB -P CHM137
#BSUB -W 02:00
#BSUB -nnodes 1
#BSUB -alloc_flags gpumps
#BSUB -J RunSpinifel
#BSUB -o RunSpinifel.%J
#BSUB -e RunSpinifel.%J

t_start=`date +%s`

# spinifel
source setup/env.sh

#jsrun -n 1 -a 1 -c 42 -g 6 -b none -d packed legion_python -ll:py 1 -ll:pyomp 36 -ll:csize 8192 -ll:show_rsrv -lg:prof 1 -lg:prof_logfile prof_%.gz legion_main.py --default-settings=summit_legion.toml --mode=legion # Legion

#jsrun -n 2 -a 1 -c 21 -g 3 -b none -d packed legion_python -ll:py 1 -ll:pyomp 18 -ll:csize 8192 -ll:show_rsrv -lg:prof 2 -lg:prof_logfile prof_%.gz legion_main.py --default-settings=summit_legion.toml --mode=legion

jsrun -n 6 -a 1 -c 7 -g 1 -b rs -d packed legion_python -ll:py 1 -ll:pyomp 4 -ll:csize 8192 -ll:show_rsrv -lg:prof 6 -lg:prof_logfile prof_%.gz legion_main.py --default-settings=summit_legion.toml --mode=legion # Legion


t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
