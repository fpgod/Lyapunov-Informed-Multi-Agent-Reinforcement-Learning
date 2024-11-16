#!/bin/sh
env="mujoco"
algo="mappo"
seed=0
kl_threshold=1e-4

python train/train_robots.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --seed_specify \
    --seed ${seed} \
    --critic_lr 5e-3 \
    --oracle_critic_lr 5e-3 \
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
    --adv_agent_ids 0 \
    --checkpoint_path "/root/old_robots_adv/scripts/results/mujoco_olld/lr0.0001/robots/0/run1/models/4804000" \
    --eval_episode 16 \
    --eval_internal 5 \
    --attack_use_recurrent \
    --use_nni
