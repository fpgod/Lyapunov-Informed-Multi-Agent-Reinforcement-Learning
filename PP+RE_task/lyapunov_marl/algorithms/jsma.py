import torch
import torch.nn as nn
from algorithms.utils.util import check

class JSMA():
    def __init__(self, args, victim, attack, device=torch.device("cpu")):
        self.theta = args.theta
        self.iter = args.iteration
        self.attack = attack
        self.victim = victim
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
    @torch.enable_grad()
    def forward(self, obs, rnn_states_actor, rnn_states_actor_victim, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_actor_victim = check(rnn_states_actor_victim).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        target = self.attack.actor.get_logits(obs, rnn_states_actor, masks, available_actions)

        n_action = target.size(-1)
        n_state = obs.size(-1)
        bs = obs.size(0)

        if self.args.env_name == "StarCraft2":
            target = target.argmax(dim=-1).clone().detach()

        domain = torch.ones_like(obs).int()

        for j in range(self.iter):
            obs = obs.requires_grad_()
            outputs = self.victim.actor.get_logits(obs, rnn_states_actor_victim, masks, available_actions)

            grads = []
            for i in range(n_action):
                grad = torch.autograd.grad(outputs[:, i].sum(), obs, retain_graph=True)[0]
                grads.append(grad.clone().detach())
            grads = torch.stack(grads)

            grads_target = grads[target, range(bs), :]
            grads_other = grads.sum(dim=0) - grads_target

            target_sum = grads_target.reshape(-1, n_state, 1) + grads_target.reshape(-1, 1, n_state)
            other_sum = grads_other.reshape(-1, n_state, 1) + grads_other.reshape(-1, 1, n_state)
            scores = -target_sum * other_sum

            scores_mask = scores.gt(0) & domain.reshape(-1, n_state, 1).ne(0) & domain.reshape(-1, 1, n_state).ne(0)
            scores_mask[:, range(n_state), range(n_state)] = 0

            valid = scores_mask.reshape(-1, n_state * n_state).any(dim=1, keepdim=True).int()

            scores = scores_mask.int() * scores
            best = torch.argmax(scores.reshape(-1, n_state * n_state), dim=-1)

            p1 = best // n_state
            p2 = best % n_state

            target_sum = target_sum * scores_mask
            sign = target_sum[range(bs), p1, p2].reshape(-1, 1).sign()
            
            p1 = torch.nn.functional.one_hot(p1.long(), num_classes=n_state)
            p2 = torch.nn.functional.one_hot(p2.long(), num_classes=n_state)

            domain = domain - valid * (p1 + p2)
            obs = obs.clone().detach() + sign * valid * (p1 + p2) * self.theta
            obs = torch.clamp(obs, min=-1, max=1).clone().detach()

        return obs.cpu().numpy()
