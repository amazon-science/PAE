import boto3
import json
import random
from botocore.config import Config
import torch
import os
import base64
from tqdm import tqdm

from PIL import Image
import base64
import io
import concurrent.futures

import random

# ========== ALL ARGUMENTS TO PLAY WITH
tasks = ["Allrecipes", "Amazon", "Apple", "Arxiv", "Github", "EPSN", "Coursera",
         "Cambridge Dictionary", "BBC News", "Google Map", "Google Search", "HuggingFace", "Wolfram Alpha"]
task_urls = {
    "Allrecipes": "https://www.allrecipes.com/",
    "Amazon": "https://www.amazon.com/",
    "Apple": "https://www.apple.com/",
    "Arxiv": "https://arxiv.org/",
    "Github": "https://github.com/",
    "ESPN": "https://www.espn.com/",
    "Coursera": "https://www.espn.com/",
    "Cambridge Dictionary": "https://dictionary.cambridge.org/",
    "BBC News": "https://bbcnews.com/",
    "Google Map": "https://www.google.com/maps",
    "Google Search": "https://www.google.com/",
    "HuggingFace": "https://huggingface.co/",
    "Wolfram Alpha": "https://www.wolframalpha.com/"
}
num_tasks_per_request = 25
total_tasks_per_website = 25
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
region_name = "us-east-1"
save_path = "/mnt/efs/yifeizhou/data/webvoyager_release_test.jsonl"
# ========== END OF ALL ARGUMENTS TO PLAY WITH




results = []
def invoke_model(request_body):
    client = boto3.client(service_name="bedrock-runtime", region_name=region_name)
    retry_time = 0
    while True:
        try:
            return client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
        except Exception as e:
            print(e)
            retry_time += 1
            if retry_time > 1:
                return None

def save_tasks(results, path, task_urls):
    raw_actions = [result['content'][0]['text'] for result in results]
    actions = [result['content'][0]['text'].split("Output:\n")[1] for result in results if "Output:\n" in result['content'][0]['text']]


    new_tasks = []
    for action in actions:
        for line in action.split("\n"):
            try:
                if line:
                    new_tasks.append(json.loads(line))
            except:
                pass

    # new_tasks = new_tasks + existing_task
    with open(path, "w") as f:
        # Iterate over the list of JSON objects
        for obj in new_tasks:
            try:
                if not isinstance(obj, dict):
                    continue
                web_name = obj["web_name"]
                web_url = task_urls[web_name]
                obj["web"] = web_url
                # Convert the JSON object to a string and write it to the file followed by a newline
                f.write(json.dumps(obj) + "\n")
            except:
                pass

boto_config = Config(read_timeout=100)

# from datasets import load_dataset

# ds = load_dataset("proj-persona/PersonaHub", "persona")


print("Number of websites: ", len(tasks))
for web_name in tqdm(tasks):
    request_bodies = []
    for _ in range(total_tasks_per_website//num_tasks_per_request):
        query = f"""
        {str({"web_name": "Apple", "id": "Apple--40", "ques": "Find the pricing and specifications for the latest Mac Studio model, including the available CPU and GPU options.", "web": "https://www.apple.com/"})}
        We are training a model to navigate the web. We need your help to generate instructions. With the examples provided above, please give {num_tasks_per_request} more example tasks for the model to learn from in the domain of {web_name}.
        You should imagine tasks that are likely proposed by a most likely user of this website. 
        YOU SHOULD MAKE USE OF THE DEMOS PROVIDES TO GENERATE TASKS, SO THAT YOUR TASKS ARE REALISTIC AND RELEVANT TO THE WEBSITE.
        Please follow the corresponding guidelines:
        1)First output your thoughts first on how you should come up with diverse tasks that examine various capabilities on the particular website, and how these tasks reflect the need of the potential user. Then you should say 'Output:\n' and then followed by the outputs STRUCTURED IN JSONL FORMAT. You should not say anything else in the response. 
        2)PLEASE MAKE SURE TO HAVE {num_tasks_per_request} examples in the response!!!
        3)Your proposed tasks should be DIVERSE AND COVER A WIDE RANGE OF DIFFERENT POSSIBILITIES AND DIFFICULTY in the domain of {web_name}. Remember, your job is to propose tasks that will help the model learn to navigate the web to deal with various real world requests.
        4)Your task should be objective and unambiguous. The carry-out of the task should NOT BE DEPENDENT on the user's personal information such as the CURRENT TIME OR LOCATION.
        5)You should express your tasks in as diverse expressions as possible to help the model learn to understand different ways of expressing the same task.
        6)Your tasks should be able to be evaluated OBJECTIVELY. That is, by looking at the last three screenshots and the answer provided by an agent, it should be possible to tell without ambiguity whether the task was completed successfully or not.
        7)Your tasks should require a minimum completion steps from 3 to 7 steps, your tasks should have a diverse coverage in difficulty as measured by the minimum completion step. I.E. You should propose not only tasks that may take more than 4 steps to complete but also tasks that can be completed within 3 steps.
        8)Humans should have a 100% success rate in completing the task. 
        9)Your tasks should be able to be completed without having to sign in to the website.
        """

        message = [{"role": "user", "content": [{"type": "text", "text": query}]}]

        request_bodies.append({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": message,
            "temperature": 1.0,
            "top_p": 0.9,
            })


    random.shuffle(request_bodies)
    for i in tqdm(range(0, len(request_bodies), 100)):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            jobs = []
            for request_body in request_bodies[i:i+100]:
                jobs.append(executor.submit(invoke_model,request_body))
            responses = [job.result() for job in jobs]
        results += [json.loads(response.get('body').read()) for response in responses if response is not None]
        # import IPython; IPython.embed(); exit()
        save_tasks(results, save_path, task_urls)
