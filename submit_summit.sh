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
./scripts/run_summit_mult.sh -m -N 1 -n 1 -t 1 -d 1 -g 1 -c -f

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
