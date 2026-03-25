export CUDA_VISIBLE_DEVICES=1,2
python -m train --cfg configs/deto_h2s_rvq_hierarchical_3layer_whisper.yaml  --use_gpus 1,2 --nodebug

python -m train --cfg configs/deto_h2s_rvq_hierarchical_3layer_whisper_h2s.yaml  --use_gpus 1,2 --nodebug



