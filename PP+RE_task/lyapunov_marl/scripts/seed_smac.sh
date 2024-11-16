#!/bin/sh
env="StarCraft2"
map="1m2m8m_vs_1m3m8m" # 1m2m8m_vs_1m3m8m 9m 4m_vs_3m 3s6z_vs_3s5z 3s6z
algo="mappo"
exp="shared"
seed=0
kl_threshold=0.06
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=2 python train/train_smac.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --map_name ${map} \
    --seed_specify \
    --seed ${seed} \
    --gamma 0.95 \
    --n_training_threads 32 \
    --n_rollout_threads 20 \
    --num_mini_batch 1 \
    --episode_length 160 \
    --num_env_steps 20000000 \
    --ppo_epoch 5 \
    --stacked_frames 1 \
    --kl_threshold ${kl_threshold} \
    --use_value_active_masks \
    --use_eval \
    --add_center_xy