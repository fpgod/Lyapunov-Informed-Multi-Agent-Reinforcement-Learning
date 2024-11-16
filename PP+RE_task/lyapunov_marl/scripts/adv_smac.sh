#!/bin/sh
env="StarCraft2"
map="11m"
algo="mappo"
adv_algo="mappo_advinf"
exp="temp"
seed=0
kl_threshold=0.06
lr=5e-4
echo "env is ${env}, map is ${map}, algo is ${algo}, adv_algo is ${adv_algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python train/train_smac.py \
    --reward_only_positive \
    --reward_defeat -200 \
    --reward_negative_scale 1 \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${adv_algo}_${exp}_${lr} \
    --map_name ${map} \
    --seed_specify \
    --seed ${seed} \
    --gamma 0.95 \
    --n_training_threads 32 \
    --n_rollout_threads 20 \
    --num_mini_batch 1 \
    --episode_length 160 \
    --num_env_steps 5000000 \
    --ppo_epoch 5 \
    --stacked_frames 1 \
    --kl_threshold ${kl_threshold} \
    --use_value_active_masks \
    --use_eval \
    --add_center_xy \
    --adversarial \
    --adv_algorithm_name ${adv_algo} \
    --adv_agent_ids 0 \
    --checkpoint_path "/home/beihang/siminli/marl_adv/ckpt/smac/11m/victim/"\
    --maa_lr ${lr} \
    --oracle_lr ${lr} \
    --oracle_critic_lr ${lr} \
    --lr ${lr} \
    --critic_lr ${lr} \
    --advinf_param 0.03 \
    --position_mask_active \
    --use_oracle_prob
