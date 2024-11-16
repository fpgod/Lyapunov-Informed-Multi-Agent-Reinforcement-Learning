#!/bin/sh
env="mujoco"
algo="mappo"
exp="robots"
seed=0
kl_threshold=1e-4
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}_lynov, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=4 python train/train_robots.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --seed_specify \
    --seed ${seed} \
    --lr 5e-6 \
    --critic_lr 5e-3 \
    --std_x_coef 1 \
    --std_y_coef 5e-1 \
    --n_training_threads 8 \
    --n_rollout_threads 4 \
    --num_mini_batch 40 \
    --episode_length 1000 \
    --num_env_steps 10000000 \
    --ppo_epoch 5 \
    --kl_threshold ${kl_threshold} \
    --use_value_active_masks \
    --use_eval \
    --add_center_xy \
    --use_state_agent
