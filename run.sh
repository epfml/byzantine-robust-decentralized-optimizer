#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 

function optimization_delta {
    COMMON="--lr 1e-3 --use-cuda --epochs 50 --batch-size 32 --max-batch-size-per-epoch 30"
    for delta in 0 0.25 0.5 0.75 1
    do
        for attack in "LF" "ALIE10" "IPM" "dissensus1.5" "BF"
        do
            python optimization_delta.py ${COMMON} -n 12 -f 1 --attack $attack --momentum 0.9 \
            --graph tcb5,1,${delta} --noniid 1 --agg "scp1" --identifier "exp" & 
            pids[$!]=$!
        done

        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done
}

function optimization_delta_plot {
    COMMON="--lr 1e-3 --use-cuda --epochs 50 --batch-size 32 --max-batch-size-per-epoch 30"
    python optimization_delta.py ${COMMON} -n 12 -f 1 --attack "BF" --momentum 0.9 \
    --graph tcb5,1,0 --noniid 1 --agg "scp1" --identifier "exp" \
    --analyze
}

function honest_majority {
    COMMON="--lr 1e-2 --use-cuda --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30"
    for agg in "scp0.1" "tm2" "rfa8" "mozi0.4,0.5"
    do
        for attack in "ALIE10" "IPM" "dissensus1.5"
        do
            python honest_majority.py ${COMMON} -n 16 -f 11 --attack ${attack} --momentum 0.9 \
            --graph mr5,1,0 --noniid 0 --agg ${agg} --identifier "exp" & 
            pids[$!]=$!
        done
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids

    for agg in "scp0.1" "tm2" "rfa8" "mozi0.4,0.5"
    do
        for attack in "ALIE10" "IPM" "dissensus1.5"
        do
            python honest_majority.py ${COMMON} -n 15 -f 10 --attack ${attack} --momentum 0.9 \
            --graph mr5,1,1 --noniid 0 --agg ${agg} --identifier "exp" & 
            pids[$!]=$!
        done
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}


function honest_majority_plot {
    COMMON="--lr 1e-2 --use-cuda --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30"
    # python honest_majority.py ${COMMON} -n 16 -f 11 --attack "LF" --momentum 0.9 \
    # --graph mr5,1,0 --noniid 0 --agg "scp1" --identifier "honest_majority" --analyze
    python honest_majority.py ${COMMON} -n 15 -f 10 --attack "LF" --momentum 0.9 \
    --graph mr5,1,1 --noniid 0 --agg "scp1" --identifier "exp" --analyze
}

function dumbbell {
    # Task: Compare aggregators on iid noniid topology
    COMMON="-n 20 -f 0 --graph dumbbell10,0,0 --attack NA --lr 0.01 --use-cuda --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30 --momentum 0.9"
    for noniid in 1 0
    do
        for agg in "tm1" "scp1" "gossip_avg" "rfa8" "mozi1,0.5"
        do
            python dumbbell.py ${COMMON} --noniid ${noniid} --agg ${agg} --identifier "exp" &
            pids[$!]=$!
        done
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

function dumbbell_plot {
    # Task: Compare aggregators on iid noniid topology 
    COMMON=" -n 20 -f 0 --graph dumbbell10,0,0 --attack NA --lr 0.01 --use-cuda --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30 --momentum 0.9"
    python dumbbell.py ${COMMON} --noniid 1 --agg "gossip_avg" --identifier "exp" --analyze
}

function dumbbell_improvement {
    # Task: noniid topology
    COMMON="-n 20 -f 0 --graph dumbbell10,0,0 --attack NA --lr 0.01 --use-cuda --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30 --momentum 0.9 --noniid 1"

    for r in "0" "1"
    do
        for agg in "tm1" "scp1" "gossip_avg" "mozi0.99,0.5" "rfa8"
        do
            python dumbbell_improvement.py ${COMMON} --graph dumbbell10,0,${r} --agg ${agg} --identifier "exp" &
            pids[$!]=$!
        done

        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done

    for r in "0" "1"
    do
        for agg in "tm1bucketing" "scp1bucketing" "gossip_avgbucketing" "mozi0.99,0.5bucketing" "rfa8bucketing"
        do
            python dumbbell_improvement.py ${COMMON} --graph dumbbell10,0,${r} --agg ${agg} --identifier "exp" &
            pids[$!]=$!
        done

        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done
}

function dumbbell_improvement_plot {
    # Task: noniid topology
    COMMON="-n 20 -f 0 --graph dumbbell10,0,0 --attack NA --lr 0.01 --use-cuda --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30 --momentum 0.9 --noniid 1"

    python dumbbell_improvement.py ${COMMON} --graph dumbbell10,0,0 --agg "gossip_avg" --identifier "exp" --analyze
}


function dumbbell_CIFAR {
    # Task: Compare aggregators on iid noniid topology
    COMMON="-n 20 -f 0 --graph dumbbell10,0,0 --attack NA --lr 0.1 --use-cuda --epochs 150 --batch-size 64 --max-batch-size-per-epoch 9999 --momentum 0.9"
    for noniid in 1 0
    do
        for agg in "tm1" "gossip_avg"  "scp10"
        do
            python dumbbell_CIFAR.py ${COMMON} --noniid ${noniid} --agg ${agg} --identifier "exp" &
            pids[$!]=$!
        done

        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done

    for noniid in 1 0
    do
        for agg in "rfa8" "mozi0.4,0.5"
        do
            python dumbbell_CIFAR.py ${COMMON} --noniid ${noniid} --agg ${agg} --identifier "exp" &
            pids[$!]=$!
        done
    done
    
    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

function dumbbell_CIFAR_plot {
    # Task: Compare aggregators on iid noniid topology 
    COMMON="-n 20 -f 0 --graph dumbbell10,0,0 --attack NA --lr 0.1 --use-cuda --epochs 150 --batch-size 64 --max-batch-size-per-epoch 9999 --momentum 0.9"

    for noniid in 1
    do
        for agg in "gossip_avg"
        do
            python dumbbell_CIFAR.py ${COMMON} --noniid ${noniid} --agg ${agg} --identifier "exp" --analyze
        done
    done
}

PS3='Please enter your choice: '
options=("debug" "optimization_delta" "optimization_delta_plot" "honest_majority" "honest_majority_plot" "dumbbell" "dumbbell_plot" "dumbbell_improvement" "dumbbell_improvement_plot" "dumbbell_CIFAR" "dumbbell_CIFAR_plot" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Quit")
            break
            ;;

        "optimization_delta")
            optimization_delta
            ;;
        "optimization_delta_plot")
            optimization_delta_plot
            ;;

        "honest_majority")
            honest_majority
            ;;

        "honest_majority_plot")
            honest_majority_plot
            ;;

        "dumbbell")
            dumbbell
            ;;

        "dumbbell_plot")
            dumbbell_plot
            ;;

        "dumbbell_improvement")
            dumbbell_improvement
            ;;

        "dumbbell_improvement_plot")
            dumbbell_improvement_plot
            ;;

        "dumbbell_CIFAR")
            dumbbell_CIFAR
            ;;
        
        "dumbbell_CIFAR_plot")
            dumbbell_CIFAR_plot
            ;;

        "debug")
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done


