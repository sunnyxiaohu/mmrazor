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


function start_rank_jobs {
    CUDA_VISIBLE_DEVICES=$1
    XPORT=$2
    START=$3
    END=$4
    CLIST=$5
    if [[ $CLIST == "None" ]]; then
        XRANGE=20
    else
        XRANGE=1
        CLIST=($CLIST)
    fi
    for XSTART in $(seq ${START} ${XRANGE} ${END}); do
        XEND=$(($XSTART + $XRANGE))
        if [[ $XEND -gt $END ]]; then
            XEND=$END
        fi
        if [[ $CLIST == "None" ]]; then
            XRANGE_STR=$XSTART-$XEND
        else
            XRANGE_STR=${CLIST[$XSTART]}
        fi
        echo $CUDA_VISIBLE_DEVICES $XPORT $XRANGE_STR ${@:6}
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        nohup python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=1 \
            --master_port=$XPORT \
            $(dirname "$0")/nas-mqbench.py \
            $CONFIG $XRANGE_STR \
            ${@:6}
    done

}


if [[ $RESUME != "None" ]]; then
    FILES=$(ls $RESUME)
    UNVIABLE_FILES=()
    # 1. check evaluated and re-added if not correct.
    for FILE in $FILES; do
        FILE_DIR=$RESUME'/'$FILE
        TRY=$(ls $FILE_DIR -l | awk '/^d/ {print $NF}')
        TRY_ARRY=(`echo $TRY`)
        # echo ${#TRY_ARRY[@]}
        if [[ ! ${#TRY_ARRY[@]} -eq 1 ]]; then
            rm -rf $FILE_DIR
            echo "Delete $FILE_DIR for multiple try experiments"
            UNVIABLE_FILES+=($FILE)
        else
            TRY_FILE=$FILE_DIR/$TRY/$TRY.json
            if [[ ! -f $TRY_FILE ]]; then
                rm -rf $FILE_DIR
                echo "Delete $FILE_DIR for $TRY_FILE is not exists"
                UNVIABLE_FILES+=($FILE)
            fi
        fi
    done
    # FULL_FILES
    for IDX in $(seq ${START} $(($END - 1))); do
        TMP=$(printf "%06d" $IDX)
        if echo "${FILES[@]}" | grep -wq "$TMP"; then
            continue
        fi
        UNVIABLE_FILES+=($TMP)
    done
    echo "There are ${#UNVIABLE_FILES[@]} indexs to be resumed."

    RANGE=${#UNVIABLE_FILES[@]}
    BIN=$(($RANGE / $GPUS))
    if [[ $RANGE -eq 0 ]]; then
        exit
    elif [[ $BIN -eq 0 ]]; then
        GPUS=1
    fi
    XSTART=0
    XEND=-1
    for DEVICE in $(seq 1 ${GPUS}); do
        VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(($DEVICE - 1))}
        XPORT=$(($PORT + $DEVICE - 1))
        XSTART=$(($XEND + 1))
        if [[ $DEVICE -eq ${GPUS} ]]; then
            XEND=$RANGE
        else
            XEND=$(($XSTART + $BIN))
        fi
        echo $DEVICE $XSTART $XEND $XPORT ${UNVIABLE_FILES[$XSTART]}
        start_rank_jobs $VISIBLE_DEVICES $XPORT $XSTART $XEND "${UNVIABLE_FILES[*]}" ${@:4} > $(date +%s)_nohup_$XSTART-$XEND.log &
    done
else
    UNVIABLE_FILES="None"
    RANGE=$(($END - $START))
    XSTART=$START
    BIN=$(($RANGE / $GPUS))
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
        start_rank_jobs $VISIBLE_DEVICES $XPORT $XSTART $XEND "None" ${@:4} > $(date +%s)_nohup_$XSTART-$XEND.log &
        XSTART=$(($XEND + 1))
    done
fi
