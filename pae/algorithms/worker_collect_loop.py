from pae.environment import batch_interact_environment
import os
import torch

def worker_collect_loop(env,\
                agent,\
                accelerator,\
                rollout_size: int = 50,\
                epochs:int = 3, \
                gamma: float = 0.9,
                save_path: str = None,
                decode_f: callable = lambda x: x,
                safe_batch_size: int = 4,
                **kwargs):
    train_trajectories = []
    assert os.path.exists(os.path.join(save_path, "../model.pt")), "Model checkpoint does not exist"
    state_dict = torch.load(os.path.join(save_path, "../model.pt"))
    if hasattr(agent.base, 'base'):
        agent.base.base.load_state_dict(state_dict)
    else:
        agent.base.load_state_dict(state_dict)
    
    agent.base = accelerator.prepare(agent.base)
    for epoch in range(epochs):
        trajectories = batch_interact_environment(agent = agent,\
                                env = env,\
                                num_trajectories= rollout_size,\
                                accelerator = accelerator,\
                                use_tqdm=True,
                                decode_f = decode_f,
                                gamma = gamma,
                                safe_batch_size = safe_batch_size,
                                iter=epoch)
        train_trajectories += trajectories
        torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))

            