import transformers
from tqdm import tqdm
from pae.models import LlavaAgent, ClaudeAgent
from pae.models.critic import TrajectoryCritic
from pae.algorithms import onpolicy_train_loop, worker_collect_loop
from pae.misc import colorful_print
from pae.environment.webgym import BatchedWebEnv
from pae.misc import colorful_print

import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import  InitProcessGroupKwargs
transformers.logging.set_verbosity_error()

import json

import random

import transformers


import accelerate
from accelerate.state import AcceleratorState
from pae.environment.webgym.utils import replace_ec2_address

def load_task_file(assets_path, task_set, task_split):
    all_tasks = []
    with open(os.path.join(assets_path, task_set + "_" + task_split + ".txt")) as fb: 
        for line in fb:
            all_tasks.append(line.replace("\n", ""))
    return all_tasks


@hydra.main(version_base=None, config_path="config/main", config_name="sft_llava")
def main(config: "DictConfig"):

    colorful_print(OmegaConf.to_yaml(config), fg='red')
    
    accelerator = accelerate.Accelerator(gradient_accumulation_steps = config.grad_accum_steps)
    #if we are using deepspeed, set microbatch size to 1
    if hasattr(AcceleratorState(), 'deepspeed_plugin') and hasattr(AcceleratorState().deepspeed_plugin, 'deepspeed_config'):
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=6000))], gradient_accumulation_steps=config.grad_accum_steps)
    device = accelerator.device
    if config.agent_name == 'llava':
        agent = LlavaAgent(policy_lm = config.policy_lm, 
                            device = device, 
                            accelerator = accelerator,
                            config = config)
    elif config.agent_name == 'claude':
        agent = ClaudeAgent(policy_lm = config.policy_lm, 
                            device = device, 
                            accelerator = accelerator,
                            config = config)
    elif config.agent_name == 'trajectory':
        agent = TrajectoryCritic(critic_lm = config.critic_lm,
                                device = device,
                                accelerator = accelerator,
                                in_dim = config.in_dim,
                                out_dim = config.out_dim)
    tasks = []
    test_tasks = []
    if config.train_tasks is not None:
        with open(config.train_tasks, 'r', encoding='utf-8') as f:
            for line in f:
                    tasks.append(json.loads(line))
    if config.test_tasks is not None:
        with open(config.test_tasks, 'r', encoding='utf-8') as f:
            for line in f:
                    if hasattr(config.env_config, "webarena_host"):
                        test_tasks.append(json.loads(replace_ec2_address(line, config.env_config.webarena_host)))
                    else:
                        test_tasks.append(json.loads(line))
    with open(config.evaluator_prompt_path, "r") as fb:
        evaluator_prompt = fb.read()

    # Create the environment
    if config.train_tasks is not None:
        env = BatchedWebEnv(tasks = tasks,
                            evaluator_prompt=evaluator_prompt,
                            download_dir=os.path.join(config.save_path, 'driver', 'download'),
                            output_dir=os.path.join(config.save_path, 'driver', 'output'),
                        **config.env_config)
    else:
        env = None
    if config.test_tasks is not None:
        test_env = BatchedWebEnv(tasks = test_tasks,
                             evaluator_prompt=evaluator_prompt,
                        download_dir=os.path.join(config.save_path, 'test_driver', 'download'),
                        output_dir=os.path.join(config.save_path, 'test_driver', 'output'),
                        batch_size=8*(len(test_tasks)//8),
                        max_iter=config.env_config.max_iter,
                        use_webarena_eval=True,
                        webarena_host=config.env_config.webarena_host if hasattr(config.env_config, "webarena_host") else "",
                        random_task=False,
                        ssh_key_path=config.env_config.ssh_key_path if hasattr(config.env_config, "ssh_key_path") else "/home/ubuntu/.ssh/id_rsa",
                        verbose=config.env_config.verbose if hasattr(config.env_config, "verbose") else False,)
    else:
        test_env = None
        
    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key = config.wandb_key)
        wandb.init(project = config.project_name, name = config.run_name, entity=config.entity_name, config = OmegaConf.to_container(config, resolve = True))
    if config.parallel_option in ["single", "host"]:
        onpolicy_train_loop(env = env,
                        eval_env = test_env,
                        agent = agent,
                        accelerator = accelerator,
                        **config)
    elif config.parallel_option == "worker":
        worker_collect_loop(env = env,
                        eval_env = test_env,
                        agent = agent,
                        accelerator = accelerator,
                        **config)
    
if __name__ == "__main__":
    main()
