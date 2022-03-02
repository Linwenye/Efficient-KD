# ------- Specification -------
#
# $1: GPU number
# $2: model
# $3: dataset
# $4: comment
#
# -----------------------------

CHINESE_DATASETS=(texsmart pku meeting)
# ENGLISH_DATASETS=(conll-03 conll-2000 ontonotes-5)

IS_CHINESE_TASK=""
for e in ${CHINESE_DATASETS[@]}
do
  if [ "$3" == "$e" ]; then
    IS_CHINESE_TASK="--chinese"
  fi
done

if [ ! -d logs/${3}/${4} ]; then
  mkdir -p logs/${3}/${4}
fi

RESOURCE_DIR=resource
RANDOM_STATE=0
EPOCH_NUM=15
BATCH_SIZE=8
HIDDEN_DIM=300
DROPOUT_RATE=0.3

CUDA_VISIBLE_DEVICES=$1 python3 -u distil.py ${IS_CHINESE_TASK} \
    --model ${2} \
    --data_dir datasets/benchmarks/${3} \
    --save_dir dump/${3}/${4}/default \
    --teacher_dir dump/${3}/try/default \
    --resource_dir ${RESOURCE_DIR} \
    --random_state ${RANDOM_STATE} \
    --epoch_num ${EPOCH_NUM} \
    --batch_size ${BATCH_SIZE} \
    --hidden_dim ${HIDDEN_DIM} \
    --dropout_rate ${DROPOUT_RATE} > logs/${3}/${4}/default.log
