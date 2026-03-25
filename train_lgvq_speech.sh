export CUDA_VISIBLE_DEVICES=1,2
export TORCH_CUDNN_V8_API_DISABLED=1
# Speech-driven LGVQ - How2Sign
python -m train --cfg configs/deto_h2s_rvq_3_lgvq_whisper.yaml --use_gpus 0,1 --nodebug

# Speech-driven LGVQ - Youtube3Då
python -m train --cfg configs/deto_h2s_rvq_3_lgvq_whisper_ytb.yaml --use_gpus 0,1 --nodebug
x