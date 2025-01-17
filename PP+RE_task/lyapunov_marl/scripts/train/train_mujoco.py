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
from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
"""Train script for Mujoco."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                env_args = {"scenario": all_args.scenario,
                            "agent_conf": all_args.agent_conf,
                            "agent_obsk": all_args.agent_obsk,
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                env_args = {"scenario": all_args.scenario,
                            "agent_conf": all_args.agent_conf,
                            "agent_obsk": all_args.agent_obsk,
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
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

def add_env_args_to_parser(args, parser, num_agents, num_adv_agents, device, n_actions):

    parser.add_argument("--num_agents", type=int, default=num_agents)
    parser.add_argument("--num_adv_agents", type=int, default=num_adv_agents)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--n_actions", type=str, default=n_actions)
    all_args = parser.parse_known_args(args)[0]
    return all_args

def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='Hopper-v2', help="Which mujoco task to run on")
    parser.add_argument('--agent_conf', type=str, default='3x1')
    parser.add_argument('--agent_obsk', type=int, default=0)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--use_single_network", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]

    return all_args


env_dict = {
    "halfcheetah-6x1": ["HalfCheetah-v2", "6x1", "../ckpt/mujoco/HalfCheetah-v2_6x1/victim"],
    "halfcheetah-2x3": ["HalfCheetah-v2", "2x3", "../ckpt/mujoco/HalfCheetah-v2_2x3/victim"],
    "walker-2x3": ["Walker2d-v2", "2x3", "../ckpt/mujoco/Walker2d-v2_2x3/victim"],
    "walker-6x1": ["Walker2d-v2", "6x1", "../ckpt/mujoco/Walker2d-v2_6x1/victim"],
    "ant-4x2": ["Ant-v2", "4x2", "../ckpt/mujoco/Ant-v2_4x2/victim"],
    "humanoid-17x1": ["Humanoid-v2", "17x1", "../ckpt/mujoco/Humanoid-v2_17x1/victim"],
    "humanoidstandup-17x1": ["HumanoidStandup-v2", "17x1", "../ckpt/mujoco/HumanoidStandup-v2_17x1/victim"],
    "manyagent_swimmer-4x2": ["manyagent_swimmer", "4x2", "../ckpt/mujoco/manyagent_swimmer_4x2/victim"],
}


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # automated param
    if all_args.use_nni:
        import nni
        params = nni.get_next_parameter()
        env_name = params["env"]
        all_args.scenario = env_dict[env_name][0]
        all_args.agent_conf = env_dict[env_name][1]
        all_args.checkpoint_path = env_dict[env_name][2]
        all_args.adv_algorithm_name = params["algorithm"]
        all_args.seed = params["seed"]
        all_args.lr = round(params["lr"], 6)
        all_args.maa_lr = round(params["lr"], 6)
        all_args.oracle_lr = round(params["lr"], 6)
        all_args.advinf_param = round(params["advinf_param"], 2)
        all_args.reward_ctrl_param = params["ctrl"]
        all_args.epsilon_state = round(params["epsilon_state"], 2)
        all_args.epsilon_action = round(params["epsilon_action"], 2)
        all_args.std_x_coef = params["std_coef"]
        all_args.std_y_coef = params["std_coef"]
        if params["algorithm"] == "mappo_usenix":
            all_args.experiment_name = all_args.experiment_name + "ctrl{}_lr{}_state{}_action{}_coef{}".format(params["ctrl"], round(params["lr"], 6), round(params["epsilon_state"], 2), round(params["epsilon_action"], 2), params["std_coef"])
        elif params["algorithm"] == "mappo_advinf":
            all_args.experiment_name = all_args.experiment_name + "ctrl{}_lr{}_param{}_coef{}".format(params["ctrl"], round(params["lr"], 6), round(params["advinf_param"], 2), params["std_coef"])
        else:
            all_args.experiment_name = all_args.experiment_name + "ctrl{}_lr{}_coef{}".format(params["ctrl"], round(params["lr"], 6), params["std_coef"])
        # if end
        if params["algorithm"] == "mappo_advinf":
            if all_args.advinf_param < 0.001:
                exit()
            if all_args.epsilon_state > 0 or all_args.epsilon_action > 0:
                exit()
        elif params["algorithm"] == "mappo_usenix":
            if all_args.advinf_param > 0:
                exit()
            if all_args.epsilon_state < 0.001 or all_args.epsilon_action < 0.001:
                exit()
        else:
            if all_args.advinf_param > 0 or all_args.epsilon_state > 0 or all_args.epsilon_action > 0:
                exit()

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
                       0] + "/results") / all_args.env_name / f"{all_args.scenario}_{all_args.agent_conf}" / all_args.adv_algorithm_name / all_args.experiment_name / str(all_args.seed)
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
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    num_adv_agents = len(all_args.adv_agent_ids)
    all_args = add_env_args_to_parser(args, parser, num_agents, num_adv_agents, device, envs.n_actions)

    if all_args.use_nni:
        env_name = params["env"]
        all_args.scenario = env_dict[env_name][0]
        all_args.agent_conf = env_dict[env_name][1]
        all_args.checkpoint_path = env_dict[env_name][2]
        all_args.adv_algorithm_name = params["algorithm"]
        all_args.seed = params["seed"]
        all_args.lr = round(params["lr"], 6)
        all_args.maa_lr = round(params["lr"], 6)
        all_args.oracle_lr = round(params["lr"], 6)
        all_args.advinf_param = round(params["advinf_param"], 2)
        all_args.reward_ctrl_param = params["ctrl"]
        all_args.epsilon_state = round(params["epsilon_state"], 2)
        all_args.epsilon_action = round(params["epsilon_action"], 2)
        all_args.std_x_coef = params["std_coef"]
        all_args.std_y_coef = params["std_coef"]
        if params["algorithm"] == "mappo_usenix":
            all_args.experiment_name = all_args.experiment_name + "ctrl{}_lr{}_state{}_action{}_coef{}".format(params["ctrl"], round(params["lr"], 6), round(params["epsilon_state"], 2), round(params["epsilon_action"], 2), params["std_coef"])
        elif params["algorithm"] == "mappo_advinf":
            all_args.experiment_name = all_args.experiment_name + "ctrl{}_lr{}_param{}_coef{}".format(params["ctrl"], round(params["lr"], 6), round(params["advinf_param"], 2), params["std_coef"])
        else:
            all_args.experiment_name = all_args.experiment_name + "ctrl{}_lr{}_coef{}".format(params["ctrl"], round(params["lr"], 6), params["std_coef"])
        # if end
        if params["algorithm"] == "mappo_advinf":
            if all_args.advinf_param < 0.001:
                exit()
            if all_args.epsilon_state > 0 or all_args.epsilon_action > 0:
                exit()
        elif params["algorithm"] == "mappo_usenix":
            if all_args.advinf_param > 0:
                exit()
            if all_args.epsilon_state < 0.001 or all_args.epsilon_action < 0.001:
                exit()
        else:
            if all_args.advinf_param > 0 or all_args.epsilon_state > 0 or all_args.epsilon_action > 0:
                exit()

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
        from runners.adv_shared.mujoco_runner import MujocoRunner as Runner
    else:
        if all_args.separate_policy:
            print("Use separated policy")
            from runners.separated.mujoco_runner import MujocoRunner as Runner
        else:
            print("Use shared policy")
            from runners.shared.mujoco_runner import MujocoRunner as Runner
    
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()

    if all_args.use_nni:
        import nni
        nni.report_final_result(0)


if __name__ == "__main__":
    main(sys.argv[1:])
