import argparse

def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["happo", "hatrpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --seed_specify
            by default True Random or specify seed for numpy/torch
        --running_id <int>
            the running index of experiment (default=1)
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)

    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --separate_policy
            by default False, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_renrecurt_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default False, do not use recurrent Policy. If set, use.
        --recurrent_N <int>
            The number of recurrent layers (default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficient of weight decay (default: 0)
    
    TRPO parameters:
        --kl_threshold <float>
            the threshold of kl-divergence (default: 0.01)
        --ls_step <int> 
            the step of line search (default: 10)
        --accept_ratio <float>
            accept ratio of loss improve (default: 0.5)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  

    
    Run parameters:
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
        --model_dir <str>
            by default None. set the path to pretrained model.

    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        
    """
    parser = argparse.ArgumentParser(description='onpolicy_algorithm', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default=' ', choices=["happo", "hatrpo", "mappo"])
    parser.add_argument("--experiment_name", type=str, 
                        default="", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, 
                        default=1, help="Random seed for numpy/torch")
    parser.add_argument("--seed_specify", action="store_true",
                        default=False, help="Random or specify seed for numpy/torch")
    parser.add_argument("--running_id", type=int, 
                        default=1, help="the running index of experiment")
    parser.add_argument("--cuda", action='store_false', 
                        default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic", action='store_false', 
                        default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, 
                        default=32, help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int,
                        default=1, help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, 
                        default=1, help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, 
                        default=10e6, help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, 
                        default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    # env parameters
    parser.add_argument("--env_name", type=str, 
                        default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--separate_policy", action='store_true',
                        default=False, help='Whether agents use different policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, 
                        default=1, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, 
                        default=64, help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, 
                        default=1, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_false', 
                        default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', 
                        default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, 
                        default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, 
                        default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, 
                        default=20, help="Time length of chunks used to train a recurrent_policy")
    
    # optimizer parameters
    parser.add_argument("--lr", type=float, 
                        default=5e-4, help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, 
                        default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, 
                        default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--std_x_coef", type=float, default=1)
    parser.add_argument("--std_y_coef", type=float, default=0.5)


    # trpo parameters
    parser.add_argument("--kl_threshold", type=float, 
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--ls_step", type=int, 
                        default=10, help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float, 
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, 
                        default=15, help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss", action='store_false', 
                        default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, 
                        default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, 
                        default=1, help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, 
                        default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm", action='store_false', 
                        default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, 
                        default=10.0, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', 
                        default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks", action='store_false', 
                        default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks", action='store_false', 
                        default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, 
                        default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    parser.add_argument("--save_interval", type=int, 
                        default=200, help="time duration between contiunous twice models saving.")
    parser.add_argument("--log_interval", type=int, 
                        default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', 
                        default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, 
                        default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, 
                        default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', 
                        default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', 
                        default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, 
                        default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, 
                        default=0.1, help="the play interval of each rendered image in saved video.")

    parser.add_argument("--use_nni", action="store_true",
                        default=False, help="by default False, whether to use nni to search parameters.")


    # adversarial arguments
    parser.add_argument("--evaluate", action="store_true",
                        default=False, help="by default False, do training process. if set, only evaluate instead.")
    parser.add_argument("--save_replay", action="store_true",
                        default=False, help="by default False, do not save replay. if set, save SC2Replay file.")
    parser.add_argument("--adversarial", action="store_true",
                        default=False, help="by default, do not train adversarial policy. if set, train adversarial.")
    parser.add_argument("--adv_algorithm_name", type=str, 
                    default='', choices=["mappo_iclr", "mappo_usenix", "mappo_icml", "mappo_advinf", "mappo_fgsm", "mappo_jsma"])               
    parser.add_argument("--adv_agent_ids", type=int, nargs="+",
                        default=[0], help="by default 0. the ids of adversarial agents.")
    parser.add_argument("--checkpoint_path", type=str, 
                    default=None, help="by default None. set the path to victim model.")
    parser.add_argument("--adv_checkpoint_path", type=str,
                        default=None, help="by default None. set the path to adversarial model.")
    parser.add_argument("--reward_ctrl_param", type=float,
                        default=0, help="the coefficient of reward_ctrl. Default is 0, which means not to cal reward ctrl.")            

    # usenix arguments
    parser.add_argument("--state_lr", type=float, 
                        default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--action_lr", type=float, 
                        default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--epsilon_state", type=float, 
                        default=0.1, help='epsilon of state transition model')
    parser.add_argument("--epsilon_action", type=float, 
                        default=0.1, help='epsilon of action transition model')

    # advinf arguments
    parser.add_argument("--maa_lr", type=float,
                        default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--oracle_lr", type=float, 
                        default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--oracle_critic_lr", type=float, 
                        default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--attack_use_naive_recurrent", action='store_true',
                        default=False, help='if maa and oracle use recurrent policy')
    parser.add_argument("--attack_use_recurrent", action='store_true',
                        default=False, help='if maa and oracle use recurrent policy')
    parser.add_argument("--advinf_param", type=float,
                        default=0.03, help='if maa and oracle use recurrent policy')
    parser.add_argument("--position_mask_active", action="store_true",
                        default=False, help='if active mask replace position mask')
    parser.add_argument("--use_oracle_prob", action="store_true",
                        default=False, help='if advinf use oracle probs')

    # state arguments
    parser.add_argument("--state_adv_mode", action="store_true",
                        default=False, help='if use fgsm/jsma in training process')
    parser.add_argument("--alpha", type=float, default=0.4, help='fgsm step size')
    parser.add_argument("--epsilon", type=float, default=1., help='fgsm max norm')
    parser.add_argument("--iteration", type=int, default=30, help='fgsm/jsma iterations')
    parser.add_argument("--theta", type=float, default=0.9, help='jsma step size')
    
    parser.add_argument("--maa_mode", type=str,
                        default='prob', help="active in mujoco only.")
    
    return parser
