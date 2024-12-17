from pae.misc import colorful_print
import threading
import os
import torch
import time


import subprocess
def save_model(accelerator, agent, save_path):
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


def run_command(worker_ip, worker_username, worker_run_path, timeout):
    command = f"cd {worker_run_path}/scripts && conda activate base && TOKENIZERS_PARALLELISM=false accelerate launch --config_file config/accelerate_config/ddp.yaml run.py --config-name worker_llava && exit"
    ssh_command = f"""ssh -tt {worker_username}@{worker_ip} << EOF 
{command}
EOF
"""
    try:
        result = subprocess.run(ssh_command, shell=True, timeout=timeout, check=True)
        if result.returncode == 0:
            colorful_print(f"Command finished successfully on {worker_ip}", fg='green')
        else:
            colorful_print(f"Command failed on {worker_ip} with return code {result.returncode}", fg='red')
    except subprocess.TimeoutExpired:
        colorful_print(f"Command on {worker_ip} timed out", fg='red')
    except subprocess.CalledProcessError as e:
        colorful_print(f"Command on {worker_ip} failed with error: {e}", fg='red')
    except Exception as e:
        colorful_print(f"An unexpected error occurred on {worker_ip}: {e}", fg='red')


def remote_collect_trajectories(save_path, 
                                worker_ips, 
                                worker_username,
                                worker_run_path,
                                host_run_path,
                                agent,
                                accelerator,
                                train_tasks,
                                test_tasks,
                                evaluator_prompt_path,
                                timeout=10800 # 3 hours for remote data collection to timeout
                                ):
    # add all workers into known hosts if not already
    colorful_print("Adding all workers to known hosts", fg='green')
    for worker_ip in worker_ips:
        print("worker_ip", worker_ip)
        os.system(f"ssh-keyscan -H {worker_ip} >> ~/.ssh/known_hosts")
    # kill all processes
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'pkill -U {worker_username}'")
    time.sleep(5)
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'skill -u {worker_username}'")
    # os.system(f"ssh {worker_username}@{worker_ip} 'sudo mount -t efs -o tls fs-d29de07b:/ /mnt/efs/'")
    time.sleep(5)
    # remove all temp files
    for worker_ip in worker_ips:
        if os.path.exists(f"{save_path}/{worker_ip}/train_trajectories.pt"):
            os.system(f"rm {save_path}/{worker_ip}/train_trajectories.pt")
        # os.system(f"rm -rf {save_path}/{worker_ip}")
    time.sleep(5)

    # copying the agent to all remote workers
    colorful_print("Saving the current trainer", fg='green')
    save_model(accelerator, agent, save_path)
    colorful_print("Copying the current trainer to all workers", fg='green')
    # copy the config arguments to all remote workers
    for worker_ip in worker_ips:
        os.system(f"scp -r {host_run_path}/scripts/config/main/worker_llava.yaml {worker_username}@{worker_ip}:{worker_run_path}/scripts/config/main")
        command = f'echo -e "\nsave_path: {save_path}/{worker_ip}" >> {worker_run_path}/scripts/config/main/worker_llava.yaml'
        os.system(f"ssh {worker_username}@{worker_ip} '{command}'")
        command = f'echo -e "\nevaluator_prompt_path: {evaluator_prompt_path}" >> {worker_run_path}/scripts/config/main/worker_llava.yaml'
        os.system(f"ssh {worker_username}@{worker_ip} '{command}'")
        command = f'echo -e "\ntrain_tasks: {train_tasks}" >> {worker_run_path}/scripts/config/main/worker_llava.yaml'
        os.system(f"ssh {worker_username}@{worker_ip} '{command}'")
        command = f'echo -e "\ntest_tasks: {test_tasks}" >> {worker_run_path}/scripts/config/main/worker_llava.yaml'
        os.system(f"ssh {worker_username}@{worker_ip} '{command}'")
    
    # parallely execute this command in all remote workser and wait for the command to finish
    threads = []
    timeout = timeout
    colorful_print("Starting all trajectory collections", fg='green')
    for worker_ip in worker_ips:
        t = threading.Thread(target=run_command, args=(worker_ip, worker_username, worker_run_path, timeout))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()




#     for worker_ip in worker_ips:
#         command = f"cd {worker_run_path}/scripts && conda activate base && TOKENIZERS_PARALLELISM=false accelerate launch --config_file config/accelerate_config/ddp.yaml run.py --config-name worker_llava && exit"
#         t = threading.Thread(target=os.system, args=(f"""ssh -tt {worker_username}@{worker_ip} << EOF 
# {command}
# EOF
# """,))
#         threads.append(t)
#         t.start()
#     for t in threads:
#         t.join()
#         colorful_print("Trajectory collection finished", fg='green')
    
    # for worker_ip in worker_ips:
    #     os.system(f"scp {worker_username}@{worker_ip}:{worker_temp_path}/trajectories.pt {host_temp_path}/{worker_ip}")
    # # wait for all trajs to be scp'ed to this host machine
    # while True:
    #     if all([os.path.exists(f"{host_temp_path}/{worker_ip}") for worker_ip in worker_ips]):
    #         break
    #     time.sleep(5)

    # # load all trajs in the remote machine
    # trajectories_list = [torch.load(f"{save_path}/{worker_ip}/train_trajectories.pt") for worker_ip in worker_ips]
    # # aggregate all trajs
    # trajectories = []
    # for trajs in trajectories_list:
    #     trajectories += trajs
    
    trajectories = []
    for worker_ip in worker_ips:
        try:
            trajectories += torch.load(f"{save_path}/{worker_ip}/train_trajectories.pt")
        except Exception as e:
            print(f"Error loading trajectories from {worker_ip}")
            colorful_print(e, fg='red')
    return trajectories
