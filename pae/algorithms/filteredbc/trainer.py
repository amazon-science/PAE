import torch
import transformers
from tqdm import tqdm
import copy
import random
from torch.utils.data import DataLoader
from pae.data import DummyDataset, DummyImageDataset
def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
class BCTrainer():
    def __init__(self, agent,
                    accelerator,
                    lm_lr: float = 1e-5,
                    batch_size: int = 4,
                    max_grad_norm: float = 1.0,
                    image_use_str: bool = False):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.lm_optimizer = torch.optim.Adam(agent.base.parameters(), lr = lm_lr)
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.image_use_str = image_use_str
    
    def prepare(self):
        self.agent.base, self.lm_optimizer = self.accelerator.prepare(self.agent.base, self.lm_optimizer)

    def actor_loss(self, observation, action, **kwargs):
        # loss = plain_bc_loss(self.agent.model, self.tokenizer, observation, action)
        loss = -self.agent.get_log_prob(observation, action).mean()
        # loss = self.agent.get_loss(observation, action)
        self.accelerator.backward(loss)
        return {"bc.loss": loss.detach().cpu().item()}

    def actor_validate(self, observation, action, **kwargs):
        with torch.no_grad():
            loss = -self.agent.get_log_prob(observation, action).mean(dim = 1).mean()
            # loss = self.agent.get_loss(observation, action)
        return {"validate.bc.loss": loss.detach().cpu().item()}
        outputs = self.agent.get_action(observation)
        corrects = []
        ill_formated = 0
        wrong_actions = 0
        for output, act in zip(outputs, action):
            try:
                # corrects.append(output == act)
                result = output.split("Action: ")[1] == act.split("Action: ")[1]
                if not result:
                    wrong_actions += 1
                    print("======> Prediction")
                    print(output)
                    print("======> Ground Truth")
                    print(act)
                corrects.append(result)
            except:
                print("======> Prediction")
                print(output)
                print("======> Ground Truth")
                print(act)
                ill_formated += 1
                corrects.append(False)
        return {"validate.bc.loss": loss.detach().cpu().item(), "validate.bc.action_correct": sum(corrects) / len(corrects),
                "validate.bc.ill_formated": ill_formated/len(corrects), "validate.bc.wrong_actions": wrong_actions/len(corrects)}

    def update(self, trajectories, actor_trajectories, iter):
        self.agent.base.train()
        random.seed(iter)
        # data = sum([random.sample(trajectories, 1)[0] for _ in range(actor_trajectories)], [])
        data = sum(random.sample(trajectories, min(actor_trajectories, len(trajectories))), [])
        dataloader = DataLoader(DummyImageDataset(data, self.image_use_str), batch_size=self.batch_size, shuffle=True, num_workers=8)
        dataloader = self.accelerator.prepare(dataloader)
        info = {}
        info_list = []
        for sample in tqdm(dataloader, disable=not self.accelerator.is_main_process):
                with self.accelerator.accumulate(self.agent.base):
                    info_list.append(self.actor_loss(**sample))
                    # if self.accelerator.sync_gradients:
                    #     self.accelerator.clip_grad_norm_(
                    #         self.agent.base.parameters(),
                    #         self.max_grad_norm
                    #     )
                    self.lm_optimizer.step()
                    self.lm_optimizer.zero_grad()
                    # torch.cuda.empty_cache()
                    # self.accelerator.free_memory()
        info.update(dict_mean(info_list))
        torch.cuda.empty_cache()
        # self.accelerator.free_memory()
        return info
    
    def validate(self, trajectories):
        self.agent.base.eval()
        data = sum(trajectories, [])
        dataloader = DataLoader(DummyImageDataset(data, self.image_use_str), batch_size=self.batch_size, shuffle=True, num_workers=8)
        dataloader = self.accelerator.prepare(dataloader)
        info = {}
        info_list = []
        with torch.no_grad():
            for sample in tqdm(dataloader, disable=not self.accelerator.is_main_process):
                        info_list.append(self.actor_validate(**sample))
        return dict_mean(info_list)

    def save(self, path):
        self.accelerator.save_state(path)
        # torch.save({'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
        #             'critic_state_dict': self.accelerator.unwrap_model(self.agent.critic).state_dict(),
        #             'target_critic_state_dict': self.accelerator.unwrap_model(self.agent.target_critic).state_dict(),
        #             'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        #             'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path)

    def load(self, path):
        self.accelerator.load_state(path)
