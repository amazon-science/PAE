import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from utils.data import DummyDataset, DummyImageDataset
import random

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list)
            else:
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class TrajectoryCriticTrainer():
    def __init__(self, agent,
                    accelerator,
                    lm_lr: float = 1e-5,
                    batch_size: int = 4,
                    max_grad_norm: float = 1.0,):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.trajectory_critic_optimizer = torch.optim.Adam(agent.parameters(), lr = lm_lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)

    def prepare(self):
        self.agent, self.trajectory_critic_optimizer = self.accelerator.prepare(self.agent, self.trajectory_critic_optimizer)

    
    def trajectory_critic_loss(self, observation, mc_return, validation = False, **kwargs):
        with torch.autograd.set_detect_anomaly(True):
            mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent).device, dtype = self.accelerator.unwrap_model(self.agent).dtype).flatten()
            v = self.agent(observation, detach_model=False)
            regression_target = (mc_return > 0).long()
            v_loss = self.criterion(v, regression_target)
            v_acc = (v.argmax(dim = 1) == regression_target).float().mean()
            if not validation:
                self.accelerator.backward(v_loss)
            v_loss = v_loss.detach().cpu()
            v_acc = v_acc.detach().cpu()
            mc_return = mc_return.detach().cpu()
            v = self.softmax(v)[:, 1]
        info = {"trajectory.v1.loss": v_loss,\
                "trajectory.v1.acc": v_acc,\
                "trajectory.v1.mean": torch.mean(v),\
                "trajectory.v1.min": torch.min(v),\
                "trajectory.v1.max": torch.max(v),\
                "trajectory.v1.std": torch.std(v),\
                "mc_return.mean": torch.mean(mc_return),
                "mc_return.max": torch.max(mc_return),
                "mc_return.min": torch.min(mc_return),
                "mc_return.std": torch.std(mc_return),
                }
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    def update(self, trajectories, actor_trajectories, iter):
        info = {}
        info_list = []
        batch_size = 8
        with torch.autograd.set_detect_anomaly(True):
                data = [{"observation": traj[0]["observation"]["task"], "mc_return": traj[-1]["reward"]} for traj in trajectories[-actor_trajectories:]]
                dataloader = DataLoader(DummyDataset(data), batch_size=self.batch_size, shuffle=True, num_workers=8)
                dataloader = self.accelerator.prepare(dataloader)
                self.trajectory_critic_optimizer.zero_grad()
                for batch in tqdm(dataloader, disable=not self.accelerator.is_main_process):
                    with self.accelerator.accumulate(self.agent):
                        info_list.append(self.trajectory_critic_loss(**batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.trajectory_critic_optimizer.step()
        info.update(dict_mean(info_list))
        return info

    def validate(self, trajectories, actor_trajectories, iter):
        info = {}
        info_list = []
        batch_size = 8
        data = [{"observation": traj[0]["observation"]["task"], "mc_return": traj[-1]["reward"]} for traj in trajectories[-actor_trajectories:]]
        dataloader = DataLoader(DummyDataset(data), batch_size=self.batch_size, shuffle=True, num_workers=8)
        dataloader = self.accelerator.prepare(dataloader)
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=True):
                info_list.append(self.trajectory_critic_loss(validation=True, **batch))
        info.update(dict_mean(info_list))
        return info