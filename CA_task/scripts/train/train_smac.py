#!/usr/bin/env python
import sys
import os
sys.path.append("../")
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from configs.config import get_config
from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.starcraft2.smac_maps import get_map_params
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
"""Train script for SMAC."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            all_args.n_actions = env.n_actions
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--use_single_network", action='store_true', default=False)

    parser.add_argument("--reward_only_positive", action='store_false', default=True)
    parser.add_argument("--reward_defeat", type=float, default=0)
    parser.add_argument("--reward_negative_scale", type=float, default=0.5)
    parser.add_argument("--obs_last_action", action='store_true', default=False)
    parser.add_argument("--replay_dir", type=str, default='/home/beihang/siminli/marl_adv/replay/')
    parser.add_argument("--replay_prefix", type=str, default='')
    all_args = parser.parse_known_args(args)[0]

    return all_args, args

def add_env_args_to_parser(args, parser, num_agents, num_adv_agents, device, n_actions, position_pace):

    parser.add_argument("--num_agents", type=int, default=num_agents)
    parser.add_argument("--num_adv_agents", type=int, default=num_adv_agents)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--n_actions", type=str, default=n_actions)
    parser.add_argument("--position_pace", type=int, default=position_pace)
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args, args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed=all_args.seed
    else:
        all_args.seed=np.random.randint(1000,10000)
    print("seed is :",all_args.seed)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name / str(all_args.seed)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        # str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        #     all_args.user_name))
        str(all_args.algorithm_name if all_args.adv_algorithm_name == '' else all_args.adv_algorithm_name) + "@" + str(all_args.map_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # position mask
    from envs.starcraft2.smac_maps import map_param_registry
    map_info = map_param_registry[all_args.map_name]
    position_pace = 5 + map_info["unit_type_bits"] + (1 if map_info["a_race"] == "P" else 0)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = get_map_params(all_args.map_name)["n_agents"]

    # don't have to, but its really stupid not to add everything in args
    num_adv_agents = len(all_args.adv_agent_ids)
    all_args = add_env_args_to_parser(args, parser, num_agents, num_adv_agents, device, envs.n_actions, position_pace)
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.adversarial:
        print("Training adversarial policy")
        from runners.adv_shared.smac_runner import SMACRunner as Runner
    else:
        if all_args.separate_policy:
            print("Use separated policy")
            from runners.separated.smac_runner import SMACRunner as Runner
        else:
            print("Use shared policy")
            from runners.shared.smac_runner import SMACRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    
    main(sys.argv[1:])

    