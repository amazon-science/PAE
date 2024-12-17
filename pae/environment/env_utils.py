import torch
from tqdm import tqdm
import numpy as np
import accelerate
from pae.misc import colorful_print
import time

def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory

#deals with the case when the environment is done
def safe_batch_get_action(agent, observation, batch_done, safe_batch_size = 4):
    new_observations = []
    obs_idxs = []
    actions = [""]*len(observation)
    for i, done in enumerate(batch_done):
        if not done and observation[i] is not None:
            new_observations.append(observation[i])
            obs_idxs.append(i)
    if len(new_observations) > 0:
        outputs = []
        for i in range(0, len(new_observations), safe_batch_size):
            try:
                outputs += agent.get_action(new_observations[i:i+safe_batch_size])
            except Exception as e:
                #sometime it says unable to infer image channel error
                colorful_print(f"Error in getting action: {e}", "red")
                outputs += ["ERROR"]*len(new_observations[i:i+safe_batch_size])
        for i, idx in enumerate(obs_idxs):
            actions[idx] = outputs[i]
    for action, obs in zip(actions, observation):
        if obs is None:
            assert action == "", "Action should be empty, First assert"
    return actions

def batch_interact_environment(agent, env, num_trajectories,\
        accelerator, post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x, safe_batch_size = 4, gamma = 0.95, iter=0):
    """
    interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    # broadcast the batch size
    torch_bsize = torch.Tensor([0,]).to(accelerator.device)
    if accelerator.is_main_process:
        torch_bsize[0] = env.batch_size
    accelerate.utils.broadcast(torch_bsize)
    total_processes = accelerator.state.num_processes
    bsize = int(torch_bsize.item())
    assert bsize % total_processes == 0, "Batch size should be divisible by the number of processes"

    all_trajectories = []
    for num_t in tqdm(range(num_trajectories//bsize), disable = not use_tqdm):
        for k in range(5):
            try:
                done = False
                trajectories = [[] for _ in range(bsize)]
                #handle the case where the reset fails and timeouts
                reset_success = torch.Tensor([False,]).to(accelerator.device)
                accelerate.utils.broadcast(reset_success)
                # while not all(reset_success):
                    # for _ in range(5):
                    #     try:
                batch_obs = [None for _ in range(bsize)]
                if accelerator.is_main_process:
                    results = env.reset()
                    batch_obs = [r[0] if r is not None else None for r in results]
                    reset_success[0] = True
                # accelerate.utils.broadcast_object_list(batch_obs)
                accelerate.utils.broadcast(reset_success)
                            # break
                        # except Exception as e:
                        #     print(f"Error in environment reset")
                        #     print(e)
                        #     accelerate.utils.broadcast(reset_success)
                        #     continue
                batch_done = torch.Tensor([False,]*bsize).to(accelerator.device)
                accelerate.utils.broadcast(batch_done)
                # print(f"Batch {num_t} iteration {k}: done {batch_done}")
                steps = 0
                while not all(batch_done):
                    steps += 1
                    if accelerator.is_main_process:
                        start_time = time.time()
                        print(f"Environment steps {str(steps)}")
                    accelerate.utils.broadcast_object_list(batch_obs)
                    # accelerate.utils.broadcast_object_list(batch_obs)
                    if True or accelerator.is_main_process:
                        # print(f"Environment steps {str(steps)}")
                        # print("getting actions!")
                        # print("Getting Action")
                        #Use timeout because NCCL watch dog will bring down if processes do not communicate after a long time
                        start = (bsize//total_processes)*accelerator.state.process_index
                        end = (bsize//total_processes)*(accelerator.state.process_index + 1)
                        # print(f"Process {accelerator.state.process_index} start {start} end {end}")
                        actions = safe_batch_get_action(agent, batch_obs[start:end], batch_done[start:end], safe_batch_size = safe_batch_size)

                    actions = accelerate.utils.gather_object(actions)
                    assert len(actions) == bsize
                    for action, done in zip(actions, batch_done):
                        if done:
                            assert action == ""
                    accelerate.utils.broadcast(batch_done)
                    if accelerator.is_main_process:
                        # actions = [agent.get_action(obs) for obs in batch_obs]
                        # print(action)
                        # import IPython; IPython.embed(); exit(1)
                        start_step_time = time.time()
                        colorful_print(f"Start stepping the environment!", "green")
                        colorful_print(f"Time taken to get action: {start_step_time - start_time}", "green")
                        batch_return = env.step(actions)
                        colorful_print(f"Done stepping the environment!", "green")
                        colorful_print(f"Time taken to step the environment: {time.time() - start_step_time}", "green")
                        # batch_return = env.step(decode_f(action))
                        # import IPython; IPython.embed()
                        for i,result in zip(range(bsize), batch_return):
                            if result is None:
                                batch_done[i] = True
                                continue
                            next_obs, r, done, info = result
                            # print(info)
                            del batch_obs[i]["message"]
                            # batch_obs[i]["message"] = None
                            trajectories[i].append({"observation": batch_obs[i], \
                                    "next_observation": next_obs, \
                                    "reward": r, \
                                    "done": done, \
                                    "action": actions[i], \
                                    "eval_info": info['eval_info'] if info is not None and 'eval_info' in info else None, \
                                    "reference": info['reference'] if info is not None and 'reference' in info else None})
                            batch_obs[i] = next_obs
                            batch_done[i] = done
                    accelerate.utils.broadcast(batch_done)
                    # print("waiting for everyone")
                    # accelerator.wait_for_everyone()
                    # obs = next_obs
                print(f"Batch {num_t} iteration {k}: done {batch_done}")
                if accelerator.is_main_process:
                    # print(trajectories[0][-1]["next_observation"])
                    all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory), gamma=gamma))\
                                        for trajectory in trajectories]
                break
            except Exception as e:
                print(f"Error in environment interaction")
                import traceback
                print(traceback.format_exc())
                print(e)
                accelerate.utils.broadcast(torch_bsize)
                continue
    # remove the message in saved trajectories because it is really big
    for trajectory in all_trajectories:
        for frame in trajectory:
            if "message" in frame["observation"]:
                del frame["observation"]["message"]
            if "message" in frame["next_observation"]:
                del frame["next_observation"]["message"]
    return all_trajectories

