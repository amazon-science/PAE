import torch
import os
NEW_PATH = "/mnt/efs/yifeizhou/driver_cache/checkpoints/webarena_example_data" #<-- Change this to the location of the downloaded data
train_trajectories = torch.load(NEW_PATH + "/train_trajectories.pt")
for trajectory in train_trajectories:
    for frame in trajectory:
        # import IPython; IPython.embed(); exit(1)
        frame["observation"]["image"] = os.path.join(NEW_PATH, frame["observation"]["image"])
        frame["next_observation"]["image"] = os.path.join(NEW_PATH, frame["next_observation"]["image"])
        assert len(frame["observation"]["image"].split(NEW_PATH)) == 2, "Parsing failed"
        assert len(frame["next_observation"]["image"].split(NEW_PATH)) == 2, "Parsing failed"
torch.save(train_trajectories, NEW_PATH + "/train_trajectories.pt")
print("====> Processing successful")
# print(frame["observation"]["image"])git 
# import IPython; IPython.embed(); exit(1)