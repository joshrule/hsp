#!/usr/bin/env bash

# This is a script for running simulations.

function run_single() {
    OUTDIR=$1
    INC=$2
    RUN=$3
    LOG=${OUTDIR}/counter_${INC}_500_${RUN}.csv

    poetry run python main.py \
	    --max_steps 100 \
	    --num_epochs 500 \
	    --num_batches 1 \
	    --num_steps 5000 \
	    --num_threads 1 \
	    --mode ppo \
	    --gamma_r 0.99 \
	    --gamma_a 0.97 \
	    --pi_lrate 0.001 \
	    --v_lrate 0.0001 \
	    --n_v_updates 80 \
	    --n_pi_updates 80 \
	    --target_kl 0.02 \
	    --seed ${RUN} \
	    --verbose 2 > $LOG
}

export -f run_single

pushd hsp/

NUM_RUNS=5
NUM_JOBS=3
INC=$1

parallel --joblog out/counter_${INC}_500_jobs.txt --jobs $NUM_JOBS "run_single out $INC {1}" ::: `seq 1 $NUM_RUNS`
popd
