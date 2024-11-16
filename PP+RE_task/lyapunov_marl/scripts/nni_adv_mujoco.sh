#!/bin/sh
env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="6x1"
agent_obsk=1
algo="mappo"
seed=0
kl_threshold=1e-4

python train/train_mujoco.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --scenario ${scenario} \
    --agent_conf ${agent_conf} \
    --agent_obsk ${agent_obsk} \
    --seed_specify \
    --seed ${seed} \
    --lr 5e-6 \
    --critic_lr 5e-3 \
    --maa_lr 5e-6 \
    --oracle_lr 5e-6 \
    --oracle_critic_lr 5e-3 \
    --std_x_coef 1 \
    --std_y_coef 5e-1 \
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
    --eval_interval 5 \
    --eval_episode 16 \
    --attack_use_recurrent \
    --use_nni
