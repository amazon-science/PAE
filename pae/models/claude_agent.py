from pae.llava_utils import find_all_linear_names, get_llava_prompts, llava_generate, llava_evaluate
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import torch

from time import sleep

from pae.misc import get_image, pad_from_left, merge_dicts


from pae.models.claude_prompts import SYSTEM_PROMPT, SYSTEM_WEBARENA_PROMPT

import boto3
import json
from botocore.config import Config
boto_config = Config(read_timeout=10)


class ClaudeAgent(torch.nn.Module):
    def __init__(self, policy_lm, device, accelerator, config):
        super(ClaudeAgent, self).__init__()
        self.policy_lm = policy_lm
        # self.base = LlavaMistralForCausalLM.from_pretrained(config.policy_lm).to(dtype=torch.bfloat16)
        # self.image_processor = self.base.get_vision_tower().image_processor
        # self.tokenizer = AutoTokenizer.from_pretrained(config.policy_lm)
        # self.use_lora = config.use_lora
        # if self.use_lora:
        #     self.base_lora_config = LoraConfig(
        #         r=256,
        #         lora_alpha=256,
        #         target_modules=find_all_linear_names(self.base, config.train_vision),
        #         lora_dropout=0.05,
        #         bias="none",
        #         task_type="CAUSAL_LM",
        #     )
        #     self.base = get_peft_model(self.base, self.base_lora_config)

        # self.tokenizer.truncation_side = 'right'
        # self.tokenizer.padding_side = 'right'
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.device = device
        self.bs = config.safe_batch_size
        
    #     self.base = self.accelerator.prepare(self.base) 

    def get_action(self, raw_observation):
        # import IPython; IPython.embed()
        #assume that the input is either a list of observations or a collated version of observation
        if isinstance(raw_observation, list):
            observation = merge_dicts(raw_observation)
        else:
            observation = raw_observation
        
        messages = observation['message']

        request_bodys = [ {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": message,
        "temperature": 1.0,
        "top_p": 0.7,
        'system' : SYSTEM_PROMPT
        } for message in messages]

        import concurrent.futures

        def invoke_model(claude_client, request_body):
            retry_time = 0
            sleep(1)
            while True:
                try:
                    return claude_client.invoke_model(
                        modelId= "anthropic.claude-3-sonnet-20240229-v1:0",#"anthropic.claude-3-5-sonnet-20240620-v1:0",
                        body=json.dumps(request_body)
                    )
                except Exception as e:
                    retry_time += 1
                    sleep(4)
                    print(f"Failed to invoke model! Retry time: {retry_time}")
                    if retry_time > 200:
                        print("Failed to invoke model!")
                        print(e)
                        return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.bs) as executor:
            jobs = []
            for request_body in request_bodys:
                client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
                jobs.append(executor.submit(invoke_model, client, request_body))
            responses = [job.result() for job in jobs]


        results = [json.loads(response.get('body').read()) if response is not None else None for response in responses]
        actions = [result['content'][0]['text'] if result is not None else "ERROR" for result in results]

        return actions

    def get_log_prob(self, observation, actions):
        pass