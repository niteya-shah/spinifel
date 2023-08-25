#!/bin/bash
#SBATCH -A chm137
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -J SpinifelCreateDemo23Data
#SBATCH -o RunSpinifel.%J
#SBATCH -e RunSpinifel.%J

t_start=`date +%s`

# Use spinifel environment
source ../setup/env.sh


export USE_CUPY=1


file_no=${1}
pushd ../setup/skopi/examples/scripts
#srun -n1 -G1 python sim_spinifel.py -b ../input/beam/amo86615.beam -p ../input/pdb/3j03.pdb -d 128 0.08 0.2 -n 200000 -s 10 -o /lustre/orion/proj-shared/chm137/demo23/data/3j03_128x128pixels_200k_${file_no}.h5
srun -n1 -G1 python sim_spinifel.py -b ../input/beam/amo86615.beam -p ../input/pdb/3iyf.pdb -d 128 0.08 0.2 -n 200000 -s 10 -o /lustre/orion/proj-shared/chm137/demo23/data/3iyf_128x128pixels_200k_${file_no}.h5
#srun -n1 -G1 python sim_spinifel.py -b ../input/beam/amo86615.beam -p ../input/pdb/3izn.pdb -d 128 0.08 0.2 -n 20000 -s 10 -o /lustre/orion/proj-shared/chm137/demo23/data/3izn_128x128pixels_20k.h5
popd


t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
