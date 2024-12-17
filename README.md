# Proposer-Agent-Evaluator (PAE): Autonomous Skill Discovery for Foundation Model Internet Agent

<p align="center">
| <a href="https://yanqval.github.io/PAE/"><b>Website | Demo | Results</b></a> | <a href="https://www.google.com/"><b>Paper</b></a> | <a href="https://huggingface.co/yifeizhou/pae-llava-7b"><b>Checkpoints</b></a> | <a href="https://huggingface.co/datasets/yifeizhou/pae-data"><b>Data</b></a> |
</p>

---

[Yifei Zhou*](https://yifeizhou02.github.io/), Qianlan Yang*, [Kaixiang Lin](https://kaixianglin.github.io/), [Min Bai](https://www.cs.toronto.edu/~mbai/), Xiong Zhou, [Yu-Xiong Wang](https://yxw.cs.illinois.edu/),[Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), [Erran Li](https://www.cs.columbia.edu/~lierranli/)<br>
UC Berkeley, UIUC, Amazon
<br>
*Equal contribution

![method_overview](https://github.com/user-attachments/assets/f4970b10-fe68-4708-ae06-223b2c865776)



## ⚙️ Features
* **State-of-the-art open-source checkpoint for generalist VLM web agent**: follow our Quick Start guide to load this checkpoint from huggingface!
* **High performance parallel environment for open-ended VLM web agent**: our implementation of web browser environment supports parallel evaluations and data collection for up to 512 concurrent browsers on 8x40 A100 node.
* Supervised fine-tuning (SFT) pipeline for VLM web agent.
* **Distributed Online RL pipeline for VLM web agent**: our online RL pipeline iterates between data collection and policy improvement stages. It also supports distributed data collection across multiple servers.

## 🚀 Quick Start
### Install Python Packages
You need to first create a conda environment and install relevant python packages
```bash
conda create -n pae python==3.10
conda activate pae

git clone https://github.com/YifeiZhou02/llava_webagent -b release
cd llava_webagent

# Install PAE
pip install -e .

# Install LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
pip install -e ".[train]"
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

### Install Chrome
You should install google chrome and chrome driver with version 125.0.6422.141 for reproducing our results
```bash
sudo apt-get update
wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_125.0.6422.141-1_amd64.deb \
  && apt install -y /tmp/chrome.deb \
  && rm /tmp/chrome.deb

wget -O /tmp/chromedriver.zip https://storage.googleapis.com/chrome-for-testing-public/125.0.6422.141/linux64/chromedriver-linux64.zip
cd /tmp
unzip /tmp/chromedriver.zip
mv chromedriver-linux64/chromedriver /usr/local/bin
rm /tmp/chromedriver.zip
rm -r chromedriver-linux64
export PATH=$PATH:/usr/local/bin
```
Then you can verify that google chrome and chromedriver have been successfully installed with 
```bash
google-chrome --version
# Google Chrome 125.0.6422.141
chromedriver --version
# ChromeDriver 125.0.6422.141
```

### Play with the Model Yourself
```python
import pae
from pae.models import LlavaAgent, ClaudeAgent
from accelerate import Accelerator
import torch
from tqdm import tqdm
from types import SimpleNamespace
from pae.environment.webgym import BatchedWebEnv
import os
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

# ============= Instanstiate the agent =============
config_dict = {"use_lora": False, 
               "use_q4": False, # our 34b model is quantized to 4-bit, set it to True if you are using 34B model
               "use_anyres": False, 
               "temperature": 1.0, 
               "max_new_tokens": 512,
               "train_vision": False,
               "num_beams": 1,}
config = SimpleNamespace(**config_dict)

accelerator = Accelerator() 
agent = LlavaAgent(policy_lm = "yifeizhou/pae-llava-7b", # alternate models "yifeizhou/pae-llava-7b-webarena", "yifeizhou/pae-llava-34b"
                            device = accelerator.device, 
                            accelerator = accelerator,
                            config = config)

# ============= Instanstiate the environment =============
test_tasks = [{"web_name": "Google Map", 
               "id": "0",
          "ques": "Locate a parking lot near the Brooklyn Bridge that open 24 hours. Review the user comments about it.", 
          "web": "https://www.google.com/maps/"}]
save_path = "xxx"

test_env = BatchedWebEnv(tasks = test_tasks,
                        do_eval = False,
                        download_dir=os.path.join(save_path, 'test_driver', 'download'),
                        output_dir=os.path.join(save_path, 'test_driver', 'output'),
                        batch_size=1,
                        max_iter=10,)
# for you to check the images and actions 
image_histories = [] # stores the history of the paths of images
action_histories = [] # stores the history of actions

results = test_env.reset()
image_histories.append(results[0][0]["image"])

observations = [r[0] for r in results]
actions = agent.get_action(observations)
action_histories.append(actions[0])
dones = None

for _ in tqdm(range(3)):
    if dones is not None and all(dones):
        break
    results = test_env.step(actions)
    image_histories.append(results[0][0]["image"])
    observations = [r[0] for r in results]
    actions = agent.get_action(observations)
    action_histories.append(actions[0])
    dones = [r[2] for r in results]

print("Done!")
print("image_histories: ", image_histories)
print("action_histories: ", action_histories)

```

## 🖥️ Download Data
You can download all release data from huggingface, including tasks, checkpoints, and example trajectories.
```bash
git lfs install
git clone git@hf.co:datasets/yifeizhou/pae-data
```
You need to process the trajectories to contain its absolute path after downloading it if you wish to perform SFT on it. Change ```NEW_PATH``` to the absolute path to the example trajectories in ```scripts/process_trajectories.py``` and run:
```bash
python scripts/process_trajectories.py
```

## 🌎 Set Pre-trained Internet Agent to Wild!
Modify configurations: change ```save_path```, ```checkpoint_path```, and ```test_tasks``` in ```scripts/config/main/evaluate_llava_7b.yaml``` according to the specifications below.
Modify ```num_processes``` in ```scripts/config/accelerate_config``` to be the number of gpus on your machine.
Now you are ready to set your agent to the Internet! The trajectories will be saved in ```save_path```.
```bash
cd scripts
TOKENIZERS_PARALLELISM=false accelerate launch --config_file config/accelerate_config/config_zero2.yaml run.py --config-name evaluate_llava_7b_no_eval
```

### Visualize the Trajectories.
We have included a script for you to visualize the trajectories that have been generated. To use it, simply run
```bash
cd scripts
python trajectory_visualizer.py {Your previous save_path}
```

### Evaluate the Trajectories and Reproduce Results.
You might have noticed that the evaluation results are all 0. This is because the autonomous evaluator has not been setup. To setup the evaluator, you will need to get an Amazon bedrock account and update the corresponding ```aws_key_id``` and
```aws_secret_key``` with information in your account. Then you are all set for reproducing our results:
```bash
cd scripts
TOKENIZERS_PARALLELISM=false accelerate launch --config_file config/accelerate_config/config_zero2.yaml run.py --config-name evaluate_llava_7b
```


## 🧑‍🎓 Supervised Fine-Tuning Example
If you wish to perform supervised fine-tuning, we perform an example workflow. You need to replace the ```offline_data_path``` entry in ```config/main/sft_llava_7b.yaml``` with the path to which you downloaded the data.
```bash
cd scripts
TOKENIZERS_PARALLELISM=false accelerate launch --config_file config/accelerate_config/config_zero2.yaml run.py --config-name sft_llava_7b
```
On an AWS P4 instance (8xA100 40G), training + evaluation takes ~8hrs.

## 🏋️ PAE RL Training example
Now you can use the tasks that we have provided to launch an RL run to see how the agents improve on its own! Fill in ```save_path```, ```checkpoint_path```, ```train_tasks```, and ```test_tasks``` entries in ```config/main/rl_llava_7b.yaml``` in to be the path where you downloaded the data.
```bash
cd scripts
TOKENIZERS_PARALLELISM=false accelerate launch --config_file config/accelerate_config/config_zero2.yaml run.py --config-name rl_llava_7b
```
On an AWS P4 instance (8xA100 40G), each iteration of 1024 trajectories takes around 2 hours, in order to accelerate, consider using parallel data collection servers discussed below. You will need to have evaluator enabled for online RL training, feel free to edit the evaluator to call your favorite model as opposed to default Claude Sonnet 3.

## 🏟️ WebArena Setup
For WebArena experiments, you need to setup self-hosted WebArena evaluators. We strongly suggest using the pre-installed AWS [images](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#pre-installed-amazon-machine-image) from WebArena to set it up. Follow [these steps](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#pre-installed-amazon-machine-image) there to set up the server. In addition, since each evaluation run requires resetting the server to a fresh state. We have included resetting remote WebArena servers in our code. To use it, 
* Generate an ssh key and add your public key to ```~/.ssh/authorized_keys``` of the WebArena server
* Modify to [```reset.sh``` and ```setup.sh```](https://github.com/YifeiZhou02/llava_webagent/tree/release/scripts/webarena_reset_scripts) be the correct ec2 addresses, and copy them to ```~/``` of the WebArena.

Then you can test our models in WebArena!
```bash
cd scripts
TOKENIZERS_PARALLELISM=false accelerate launch --config_file config/accelerate_config/config_zero2.yaml run.py --config-name evaluate_llava_7b_webarena
```

## 📝 PAE Task Proposing Example
If you wish to go through the entire pipeline of PAE including task proposing, you can do so by playing with ```python scripts/propose_tasks_from_names_webvoyager.py```. All hyperparameters can be found at the head of this python file.

## 🤝 PAE Training with Multiple Data Collection Servers
In order to speed things up, we have provided scripts for accelerating data collection (i.e. trajectory rollouts) since this is the main bottleneck in our online RL training process. You need to follow the following steps to set up remote data collection servers. The remote data collection servers are just for inference and currently we only support trainiing on a single machine. In our main experiments, we use an AWS P4 instance (8x40GB) and 3 AWS G5 instance (8x24GB) for training LLaVa-7B.
* Make sure both remote data collection servers and the main host machine have access to the same data storage system specified in the ```save_path``` argument.
* Set up the conda environment on remote data collection servers as specified above. Note that we currently only support using the ```base``` conda environment on remote data collection servers so you need to install respective packages in the base environment on remote servers.
* Add the default ssh key of your host machine (usually ```~/id_rsa.pub``` ) to the remote data collection servers (add to ```~/.ssh/authorized_keys```).

Now you can ready to go! Just keep your remote data collection servers running and no need to run any further commands on those servers manually. Now on your host machine, simply correct the paths in ```config/main/rl_llava_7b_distributed_server.yaml``` and run:
```bash
cd scripts
TOKENIZERS_PARALLELISM=false accelerate launch --config_file config/accelerate_config/config_zero2.yaml run.py --config-name rl_llava_7b_distributed_server
```
You should be able to see results logged to your wandb if you have that enabled.

## 🗂 Specification for Configurations
* ```checkpoint_path```: The path to the checkpoint that you want the model to start from. If set to ```null```, the pre-trained VLM checkpoint from huggingface will be used.
* ```offline_data_path```: The path to provided offline data that contains ```trajectories.pt```. Any directory that contains ```trajectories.pt``` can be used.
* ```save_path```: The path to save intermediate checkpoints and interaction trajectories. The script will prioritize loading from ```save_path``` for checkpoints and data over ```checkpoint_path``` and ```offline_data_path```.
* ```policy_lm```: The name of the LLaVa model to be used, currently supporting: ```liuhaotian/llava-v1.6-mistral-7b``` and ```liuhaotian/llava-v1.6-34b```.
* ```use_lora```: A boolean flag for whether or not LoRA will be applied for the model. Default: True.
* ```train_vision```: A boolean flag for whether the parameters for the vision encoder will be updated. Default: False.
* ```algorithm```: The algorithm to be used for updating the model weights. Supporting: ```sft``` and ```filteredbc```.
* ```agent_name```: The name of the agent. Supporting ```llava``` and ```claude```.
* ```use_q4```: Whether or not to apply 4-bit quantization to the model. Default: False.
* ```use_anyres```: Whether any-resolution is turned on for LLaVa-1.6. Default: False.
* ```capacity```: Use the latest N trajectories in the replay buffer for updating the model. Usually a large number will be fine.
* ```lm_lr```: Learning rate for the model.
* ```grad_accum_steps```: Gradient Accumulation Steps.
* ```online```: A boolean flag for whether online data is collected for each gloabl iteration.
* ```epochs```: Number of global iterations, including a round of data collection if ```online```. You can set it to be a large number to see the entire learning curve on wandb.
* ```actor_epochs```: Number of epochs to update the policy at each global iteration.
* ```actor_trajectories```: During each ```actor_epoch```, the number of trajectories to be used for the update. The total number of trajectories sampled (with replacement) for updating the policy is ```actor_trajectories```*```actor_epochs```.
* ```rollout_size```: Number of trajectories to be collected at each global iteration.
* ```safe_batch_size```: The batch size for policy inference during data collection.
* ```env_config/batch_size```: Number of parallel browsing environment.
* ```env_config/max_iter```: Maximum number of environment steps for each trajectory. Default: 10.
* ```env_config/do_eval```: A boolean flag for performing evaluations during rollouts. If set to False, all trajectories will have a default reward of 0.
* ```env_config/aws_key_id```: AWS bedrock key id for quering Claude for autonomous evaluations. If set to ```null```, the script will try to infer from environment variable.
* ```env_config/aws_secret_key```: AWS bedrock secret for quering Claude for autonomous evaluations. If set to ```null```, the script will try to infer from environment variable.
* ```env_config/webarena_host```: When using WebArena, the ec2 address of the webarena server.
* ```train_tasks```: The .jsonl task file to load the tasks for online training.
* ```test_tasks```: The .jsonl task file to load the tasks for test evaluations.
* ```save_freq```: The frequency (in terms of global iterations) for saving model checkpoints. Forced to be 1 while using remote data servers.
* ```eval_freq```: The frequency to perform evaluations using test tasks.
* ```use_wandb```: A boolean flag for using Weights and Biases.
* ```parallel_options```: Supports ```single```, ```host```, and ```worker```. If remote data collection is not used, use ```single```. For remote collection, use ```host``` on the host machine. ```worker`` is used in ```worker_llava.yaml``` and got automatically passed to the remote servers.
* ```worker_ips```: A list of IP addresses of your workers (so that the host server can ssh into your remote data collection servers) if remoet data collection is used.
* ```host_run_path```: The path to this github directory on the host machine if remoet data collection is used.
* ```worker_run_path```: The path to this github directory on the worker machines if remoet data collection is used.
* ```worker_username```: The username to ssh into the remote servers from the host machine if remote data collection is used.
* ```remote_timeout```: The timeout limit for each gloabl iteration on the remote machines in seconds.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This dataset is licensed under the CC-BY-NC-4.0 License. See the [LICENSE](LICENSE.txt) file.