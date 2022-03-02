# ------- Specification -------
#
# $1: GPU number
# $2: dataset
# $3: comment
# $4: parameter name
# $5: parameter values
#
# -----------------------------

CHINESE_DATASETS=(texsmart pku)
# ENGLISH_DATASETS=(conll-03 conll-2000 ontonotes-5)

RESOURCE_DIR=resource
RANDOM_STATE=0
EPOCH_NUM=15
BATCH_SIZE=8
HIDDEN_DIM=300
DROPOUT_RATE=0.3

IS_CHINESE_TASK=""
for e in ${CHINESE_DATASETS[@]}
do
  if [ "$2" == "$e" ]; then
    IS_CHINESE_TASK="--chinese"
  fi
done

if [ ! -d logs/${2}/${3} ]; then
  mkdir -p logs/${2}/${3}
fi

OLD_IFS="$IFS"
IFS=","
choice_list=($5)
IFS="$OLD_IFS"
echo "[Hyper-parameter Tuning] name: $4, values: ${5}"

for val in ${choice_list[@]}
do
  if [ "$4" == "hd" ]; then
    HIDDEN_DIM=${val}
  elif [ "$4" == "dr" ]; then
    DROPOUT_RATE=$val
  else
    echo "Unsupported parameter: ${7}"
    exit
  fi

  CUDA_VISIBLE_DEVICES=$1 python3 -u main.py ${IS_CHINESE_TASK} \
    --data_dir datasets/benchmarks/${2} \
    --save_dir dump/${2}/${3}/${4}_${val} \
    --resource_dir ${RESOURCE_DIR} \
    --random_state ${RANDOM_STATE} \
    --epoch_num ${EPOCH_NUM} \
    --batch_size ${BATCH_SIZE} \
    --hidden_dim ${HIDDEN_DIM} \
    --dropout_rate ${DROPOUT_RATE} > logs/${2}/${3}/${4}_${val}.log
done
