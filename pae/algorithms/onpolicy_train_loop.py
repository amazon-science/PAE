from pae.environment import batch_interact_environment
import numpy as np
from pae.algorithms.filteredbc import BCTrainer
from pae.algorithms.base import BaseTrainer
from pae.algorithms.trajectory_critic import TrajectoryCriticTrainer
from pae.misc import colorful_print
import wandb
import os
import torch
import time
from pae.algorithms.parallel_utils import remote_collect_trajectories
from pae.models import ClaudeAgent
from pae.models.critic import TrajectoryCritic
import random
from multiprocessing.pool import ThreadPool
import accelerate
from .utils import calculate_upperbound,\
    filter_trajectories, rollout_statistics_by_websites,\
    clean_trajectories

def onpolicy_train_loop(env,\
                agent,\
                accelerator,\
                rollout_size: int = 50,\
                batch_size: int = 2,
                capacity: int = 500000,
                epochs:int = 3, \
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                use_wandb: bool = False,
                online: bool = False,
                eval_env = None,
                actor_epochs: int = 3,
                actor_trajectories: int = None,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                checkpoint_path: str = None,
                save_freq: int = 25,
                eval_freq: int = 10,
                decode_f: callable = lambda x: x,
                offline_data_path: str = None,
                safe_batch_size: int = 4,
                eval_at_start: bool = False,
                algorithm: str = "filteredbc",
                parallel_option: str = "single",
                worker_ips: list = [],
                worker_username: str = "ubuntu",
                worker_run_path: str = "/home/ubuntu/digirl",
                host_run_path: str = "",
                reset_server: bool = False,
                remote_timeout: int = 10800,
                train_tasks: str = '',
                test_tasks: str = '',
                evaluator_prompt_path: str = '',
                **kwargs):
    colorful_print(f"Using Algorithm {algorithm}", fg='green')
    train_trajectories = []
    all_evaluate_trajectories = []
    if os.path.exists(os.path.join(offline_data_path, 'train_trajectories.pt')):
        train_trajectories = torch.load(os.path.join(offline_data_path, 'train_trajectories.pt'))

    if isinstance(agent, ClaudeAgent):
        trainer = BaseTrainer(agent=agent,\
                            accelerator=accelerator,\
                            lm_lr = lm_lr,\
                            max_grad_norm=max_grad_norm)
    elif isinstance(agent, TrajectoryCritic):
        trainer = TrajectoryCriticTrainer(agent=agent,\
                            accelerator=accelerator,\
                            lm_lr = lm_lr,\
                            max_grad_norm=max_grad_norm)
    else:
        trainer = BCTrainer(agent=agent,\
                            accelerator=accelerator,\
                            lm_lr = lm_lr,\
                            batch_size = 1,\
                            max_grad_norm=max_grad_norm,
                            image_use_str=True) #isinstance(agent, InternLMAgent))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(os.path.join(save_path, 'train_trajectories.pt')):
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        
    if os.path.exists(os.path.join(save_path, "model.pt")):
        colorful_print("=====>Loading from saved_path checkpoint", fg = 'green')
        state_dict = torch.load(os.path.join(save_path, "model.pt"))
        if hasattr(agent.base, 'base'):
            agent.base.base.load_state_dict(state_dict)
        else:
            agent.base.load_state_dict(state_dict)
    else:
        if checkpoint_path is not None and os.path.exists(os.path.join(checkpoint_path, 'model.pt')):
            colorful_print("=====>Loading from checkpoint_path checkpoint", fg='green')
            state_dict = torch.load(os.path.join(checkpoint_path, 'model.pt'))
            if hasattr(agent.base, 'base'):
                agent.base.base.load_state_dict(state_dict)
            else:
                agent.base.load_state_dict(state_dict)
    trainer.prepare()


    if eval_at_start:
        info = {}
        print("=====>Evaluating at start")
        if reset_server and accelerator.is_main_process:
            eval_env.reset_server()
        evaluate_trajectories = batch_interact_environment(agent = agent,\
                                env = eval_env,\
                                num_trajectories= eval_env.batch_size,\
                                accelerator = accelerator,\
                                use_tqdm=True,
                                decode_f = decode_f,
                                gamma = gamma,
                                safe_batch_size = safe_batch_size,
                                iter=0)
        if accelerator.is_main_process:
            info.update({"evaluate_rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in evaluate_trajectories]),\
                "evaluate_rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in evaluate_trajectories]),\
                "evaluate_rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in evaluate_trajectories]),\
                "evaluate_rollout.timeouts": np.mean([len(d) >= eval_env.max_iter for d in evaluate_trajectories])})
            info.update(rollout_statistics_by_websites(evaluate_trajectories, eval_env, evaluate = True))
            all_evaluate_trajectories += evaluate_trajectories
            torch.save(all_evaluate_trajectories, os.path.join(save_path, 'evaluate_trajectories.pt'))
            # torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))
        # accelerator.wait_for_everyone()
        # train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
    colorful_print(f"The upperbound performance is {str(calculate_upperbound(train_trajectories))}", fg='green')
    for epoch in range(epochs):
        info = {}
        if online:
            if reset_server and accelerator.is_main_process:
                env.reset_server()
            if parallel_option == 'host' and accelerator.is_main_process:
                pool = ThreadPool(processes=1)
                colorful_print("Starting remote data collection subroutine", fg='green')
                async_result = pool.apply_async(remote_collect_trajectories, (save_path, worker_ips, worker_username, worker_run_path, \
                    host_run_path, agent, accelerator, train_tasks, test_tasks, evaluator_prompt_path, remote_timeout))
            start_time = time.time()
            trajectories = batch_interact_environment(agent = agent,\
                                    env = env,\
                                    num_trajectories= rollout_size,\
                                    accelerator = accelerator,\
                                    use_tqdm=True,
                                    decode_f = decode_f,
                                    gamma = gamma,
                                    safe_batch_size = safe_batch_size,
                                    iter=epoch)
            colorful_print("Main thread finished collecting trajectories", fg='cyan')
            colorful_print(f"Time taken to collect trajectories: {time.time() - start_time}", fg='cyan')
            #if the remote trajectories have not finished, collect another round while waiting
            if parallel_option == 'host':
                async_ready = torch.Tensor([False,]).to(accelerator.device)
                while not async_ready[0]:
                    trajectories += batch_interact_environment(agent = agent,\
                                    env = env,\
                                    num_trajectories= env.batch_size,\
                                    accelerator = accelerator,\
                                    use_tqdm=True,
                                    decode_f = decode_f,
                                    gamma = gamma,
                                    safe_batch_size = safe_batch_size,
                                    iter=epoch)
                    if accelerator.is_main_process:
                        async_ready[0] = async_result.ready()
                    accelerate.utils.broadcast(async_ready)
            if parallel_option == 'host' and accelerator.is_main_process:                
                trajectories += async_result.get()

            if accelerator.is_main_process:
                train_trajectories += trajectories
                info.update({"iteration": epoch,\
                        "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                        "rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                        "rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                        "rollout.timeouts": np.mean([len(d) >= env.max_iter for d in trajectories]),\
                        "rollout.upperbound": calculate_upperbound(train_trajectories),\
                        "walltime": time.time()})
                info.update(rollout_statistics_by_websites(trajectories, env))
                colorful_print(f"Trajectories: {len(trajectories)}, Train Trajectories: {len(train_trajectories)}", fg='green')
                # print(trajectories[-1][-1])
                #sync the trajectories between all threads
                torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))
                torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories_backup.pt'))
            accelerator.wait_for_everyone()
            train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        for _ in range(actor_epochs):
                if algorithm == "filteredbc":
                    info.update(trainer.update(filter_trajectories(train_trajectories[-capacity:]), actor_trajectories=actor_trajectories, iter=epoch))
                elif algorithm == "sft":
                    info.update(trainer.update(clean_trajectories(train_trajectories[-capacity:]), actor_trajectories=actor_trajectories, iter=epoch))
                    # actor_trajectories = train_trajectories[-capacity:]
                # info.update(trainer.update(filter_trajectories(train_trajectories[-capacity:]), actor_trajectories=actor_trajectories, iter=epoch))


        #checkpointing
        if accelerator.is_main_process and (epoch+1)%save_freq == 0 and not isinstance(trainer, BaseTrainer) and parallel_option == "single":
                unwrapped_base = accelerator.unwrap_model(agent.base)
                if hasattr(unwrapped_base, 'base'):
                    state_dict = unwrapped_base.base.state_dict()
                else:
                    state_dict = unwrapped_base.state_dict()
                state_dict = {k: v.cpu() for k, v in state_dict.items()}
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(state_dict, os.path.join(save_path, 'model.pt'))
                agent.base.to(accelerator.device)

                
        if (epoch + 1) % eval_freq == 0:
            if eval_env is not None:
                if reset_server and accelerator.is_main_process:
                    eval_env.reset_server()
                evaluate_trajectories = batch_interact_environment(agent = agent,\
                                        env = eval_env,\
                                        num_trajectories= eval_env.batch_size,\
                                        accelerator = accelerator,\
                                        use_tqdm=True,
                                        decode_f = decode_f,
                                        gamma = gamma,
                                        safe_batch_size = safe_batch_size,
                                        iter=epoch)
                if accelerator.is_main_process:
                    info.update({"evaluate_rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in evaluate_trajectories]),\
                        "evaluate_rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in evaluate_trajectories]),\
                        "evaluate_rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in evaluate_trajectories]),\
                        "evaluate_rollout.timeouts": np.mean([len(d) >= eval_env.max_iter for d in evaluate_trajectories])})
                    info.update(rollout_statistics_by_websites(evaluate_trajectories, eval_env, evaluate = True))
                    all_evaluate_trajectories += evaluate_trajectories
                    torch.save(all_evaluate_trajectories, os.path.join(save_path, 'evaluate_trajectories.pt'))

        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
    return
    