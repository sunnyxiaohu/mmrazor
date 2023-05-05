#!/usr/bin/env bash

# set -x

CONFIG=$1
START=$2
END=$3
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RESUME=${RESUME:-"None"}


if [[ $RESUME != "None" ]]; then
    FILES=$(ls $RESUME)
    UNVIABLE_FILES=$(seq ${START} $(($END - 1)))
    for IDX in $(seq ${START} $(($END - 1))); do
        # echo $(printf "%06d" $IDX) $IDX
        UNVIABLE_FILES[$IDX]=$(printf "%06d" $IDX)
    done
    # echo $FILES
    # 1. check evaluated and re-added if not correct.
    for FILE in $FILES; do
        FILE_DIR=$RESUME'/'$FILE
        TRY=$(ls $FILE_DIR -l | awk '/^d/ {print $NF}')
        TRY_ARRY=(`echo $TRY`)
        # echo ${#TRY_ARRY[@]}
        if [[ ! ${#TRY_ARRY[@]} -eq 1 ]]; then
            echo "Delete $FILE_DIR for multiple try experiments"
        else
            TRY_FILE=$FILE_DIR/$TRY/$TRY.json
            if [[ ! -f $TRY_FILE ]]; then
                echo "Delete $FILE_DIR for $TRY_FILE is not exists"
                continue
            fi
            for IDX in $(seq 0 $((${#UNVIABLE_FILES[@]} - 1))); do
                if [[ ${UNVIABLE_FILES[$IDX]} = $FILE ]]; then
                    unset UNVIABLE_FILES[$IDX]
                fi
            done
        fi
    done
    echo "There are ${#UNVIABLE_FILES[@]} indexs to be resumed."

    RANGE=${#UNVIABLE_FILES[@]}
    BIN=$(($RANGE / $GPUS))
    XSTART=0
    XEND=0
    for DEVICE in $(seq 1 ${GPUS}); do
        VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(($DEVICE - 1))}
        XPORT=$(($PORT + $BIN * $DEVICE))
        XSTART=$XEND
        XEND=$(($START + $BIN))
        if [[ $DEVICE -eq ${GPUS} ]]; then
            XEND=$END
        else
            XEND=$(($XSTART + $BIN))
        fi

        # for IDX in $(seq ${XSTART} ${XEND}); do
        #     echo $DEVICE $(($XPORT + $IDX))  ${UNVIABLE_FILES[$IDX]}
        #     # CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES \
        #     # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        #     # nohup python -m torch.distributed.launch \
        #     #     --nnodes=$NNODES \
        #     #     --node_rank=$NODE_RANK \
        #     #     --master_addr=$MASTER_ADDR \
        #     #     --nproc_per_node=1 \
        #     #     --master_port=$(($XPORT + $IDX)) \
        #     #     $(dirname "$0")/nas-mqbench.py \
        #     #     $CONFIG ${UNVIABLE_FILES[$IDX]} \
        #     #     ${@:4} > $(date +%s)_nohup_$XSTART-$XEND.log &
        # done

    done
    exit
fi

function start_rank_jobs {
    CUDA_VISIBLE_DEVICES=$1
    XPORT=$2
    START=$3
    END=$4
    XRANGE=20
    for XSTART in $(seq ${START} ${XRANGE} ${END}); do
        XEND=$(($XSTART + $XRANGE))
        if [[ $XEND -gt $END ]]; then
            XEND=$END
        fi
        echo $CUDA_VISIBLE_DEVICES $XPORT $XSTART $XEND ${@:5}
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        nohup python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=1 \
            --master_port=$XPORT \
            $(dirname "$0")/nas-mqbench.py \
            $CONFIG $XSTART-$XEND \
            ${@:5}
    done

}


RANGE=$(($END - $START))
BIN=$(($RANGE / $GPUS))
XSTART=$START
XEND=0
for DEVICE in $(seq 1 ${GPUS}); do
    VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(($DEVICE - 1))}
    XPORT=$(($PORT + $DEVICE - 1))
    XEND=$(($START + $BIN))
    if [[ $DEVICE -eq ${GPUS} ]]; then
        XEND=$END
    else
        XEND=$(($XSTART + $BIN))
    fi
    echo $VISIBLE_DEVICES $XPORT $XSTART $XEND ${@:4}
    start_rank_jobs $VISIBLE_DEVICES $XPORT $XSTART $XEND ${@:4} > $(date +%s)_nohup_$XSTART-$XEND.log &
    XSTART=$XEND    
done
