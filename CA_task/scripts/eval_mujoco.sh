#!/bin/sh
env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="6x1"
agent_obsk=1
algo="mappo"
ctrl=0
lr=5e-4
exp="test"
adv_algo="mappo_iclr"
seed=0
kl_threshold=1e-4
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${agent_conf}_${adv_algo}_${exp} \
    --scenario ${scenario} \
    --agent_conf ${agent_conf} \
    --agent_obsk ${agent_obsk} \
    --seed_specify \
    --seed ${seed} \
    --lr ${lr} \
    --critic_lr 5e-3 \
    --maa_lr ${lr} \
    --oracle_lr ${lr} \
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
    --adv_algorithm_name ${adv_algo} \
    --adv_agent_ids 0 \
    --reward_ctrl_param ${ctrl} \
    --checkpoint_path "../ckpt/mujoco/HalfCheetah-v2_6x1/victim" \
    --adv_checkpoint_path "/home/beihang/guojun/marl_adv/scripts/results/mujoco/mujoco_seedtest/HalfCheetah-v2/mappo/6x1_shared_mappo_iclr_ctrl0_lr0.0005_param0/1/run1/models/9984000" \
    --evaluate \
    --eval_episode 8
