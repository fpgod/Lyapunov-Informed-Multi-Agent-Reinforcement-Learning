#!/bin/sh
env="mujoco"
algo="mappo"
param=0.003
lr=5e-5
exp="decay"
adv_algo="mappo_iclr"
kl_threshold=1e-4

for number in `seq 5`;
do
    CUDA_VISIBLE_DEVICES=0 python train/train_robots.py \
        --env_name ${env} \
        --algorithm_name ${algo} \
        --experiment_name ${exp} \
        --seed_specify \
        --seed ${number} \
        --lr ${lr} \
        --critic_lr 5e-3 \
        --maa_lr ${lr} \
        --oracle_lr ${lr} \
        --oracle_critic_lr 5e-3 \
        --std_x_coef 0.2 \
        --std_y_coef 0.2 \
        --n_training_threads 32 \
        --n_rollout_threads 32 \
        --num_mini_batch 40 \
        --episode_length 1000 \
        --num_env_steps 10000000 \
        --ppo_epoch 5 \
        --kl_threshold ${kl_threshold} \
        --use_value_active_masks \
        --use_eval \
        --add_center_xy \
        --use_state_agent \
        --adversarial \
        --adv_algorithm_name ${adv_algo} \
        --adv_agent_ids 0 \
        --checkpoint_path "/root/old_robots_adv/scripts/results/mujoco_olld/lr0.0001/robots/0/run1/models/4804000" \
        --eval_episode 16 \
        --eval_internal 5 \
        --advinf_param ${param} \
        --attack_use_recurrent &
done