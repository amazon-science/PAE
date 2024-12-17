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
agent = LlavaAgent(policy_lm = "yifeizhou/pae-llava-7b-webarena", #"/mnt/efs/yifeizhou/release_data/hf_checkpoints/pae-llava-7b",#"/mnt/efs/yifeizhou/release_data/hf_checkpoints/pae-llava-34b", 
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

