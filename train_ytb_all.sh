 python s10_split_dataset_by_word_distribution.py \
    --input_csv youtube3d_all_hs2.csv \
    --output_dir /data/hwu/slg_data/Youtube3D_all \
    --source_data_dir /data/hwu/youtube3d/merged_smplx_params \
    --dest_data_subdir poses \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

python compute_youtube3d_stats.py

python -m train --cfg configs/deto_h2s_rvq_3_youtube_all.yaml --use_gpus 0,1,2 --nodebug