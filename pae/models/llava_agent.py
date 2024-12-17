from pae.llava_utils import find_all_linear_names, get_llava_prompts, llava_generate, llava_evaluate
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers import AutoTokenizer
import transformers
import torch
from PIL import Image
import numpy as np

from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import LlavaForConditionalGeneration, AutoProcessor
from pae.misc import get_image, pad_from_left, merge_dicts
from transformers import LlamaForCausalLM, MistralForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# MODEL_CLASSES = {"liuhaotian/llava-v1.6-34b": LlavaLlamaForCausalLM, "liuhaotian/llava-v1.6-mistral-7b": LlavaMistralForCausalLM}

class LlavaAgent(torch.nn.Module):
    def __init__(self, policy_lm, device, accelerator, config=None,
                 use_q4 = False, use_lora = False, use_anyres = False):
        super(LlavaAgent, self).__init__()
        self.policy_lm = policy_lm
        # self.base = LlavaMistralForCausalLM.from_pretrained(config.policy_lm, load_in_8bit= True)
        if config is not None:
            use_q4 = config.use_q4
            use_lora = config.use_lora
            use_anyres = config.use_anyres
        self.use_anyres = use_anyres
        if use_q4:
            q4_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
            )
            if "7b" in policy_lm:
                self.base = LlavaMistralForCausalLM.from_pretrained(policy_lm, quantization_config=q4_config,)
            else:
                self.base = LlavaLlamaForCausalLM.from_pretrained(policy_lm, quantization_config=q4_config,)
            # self.base = MODEL_CLASSES[policy_lm].from_pretrained(policy_lm, quantization_config=q4_config,)
            self.base = prepare_model_for_kbit_training(self.base)
        else:
            if "7b" in policy_lm:
                self.base = LlavaMistralForCausalLM.from_pretrained(policy_lm)
            else:
                self.base = LlavaLlamaForCausalLM.from_pretrained(policy_lm)
                # self.base = MODEL_CLASSES[policy_lm ].from_pretrained(policy_lm).to(dtype = torch.bfloat16)
        self.model_cfg = self.base.config
        self.image_processor = self.base.get_vision_tower().image_processor
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm)
        self.use_lora = use_lora
        if self.use_lora:
            self.base_lora_config = LoraConfig(
                r=256,
                lora_alpha=256,
                target_modules=find_all_linear_names(self.base, config.train_vision if config is not None else False),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.base = get_peft_model(self.base, self.base_lora_config)
        if not use_q4:
            self.base = self.base.to(device)
        self.dtype = self.base.dtype
        self.tokenizer.truncation_side = 'right'
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.softmax = torch.nn.Softmax(dim= -1)
        self.temperature = config.temperature if config is not None else 1.0
        self.accelerator = accelerator
        self.max_new_tokens = config.max_new_tokens if config is not None else 512
        self.config = config
        self.eos_str = None
    
    
    # def prepare(self):
    #     self.base = self.accelerator.prepare(self.base) 

    def get_action(self, raw_observation):
        with torch.no_grad():
            # import IPython; IPython.embed()
            #assume that the input is either a list of observations or a collated version of observation
            if isinstance(raw_observation, list):
                observation = merge_dicts(raw_observation)
            else:
                observation = raw_observation
            prompts = get_llava_prompts(observation)
            images = observation["image"]
            # if not isinstance(images, List) and not isinstance(images, torch.Tensor) and not isinstance(images, np.ndarray):
            #     images = [get_image(images)]
            images = [get_image(image) if isinstance(image, str) else image for image in images]
            image_sizes = [image.size for image in images]
            outputs = []
            all_input_ids = []
            all_image_tensors = []
            for prompt, image in zip(prompts, images):
                if self.use_anyres:
                    image_tensor = process_images([image], self.image_processor, self.model_cfg).to(self.device, dtype=self.dtype)
                else:
                    image_tensor = self.image_processor(image, return_tensors='pt')['pixel_values'].to(self.base.device, dtype=self.dtype)
                # image_tensor = self.image_processor(image, return_tensors='pt')['pixel_values'].to(self.base.device, dtype=self.accelerator.unwrap_model(self.base).dtype)
                # print(image_tensor.shape)
                # print(len(prompt))
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
                all_input_ids.append(input_ids)
                all_image_tensors.append(image_tensor)
            input_ids = torch.Tensor(pad_from_left(all_input_ids, self.tokenizer.pad_token_id)).long().to(self.base.device)
            all_image_tensors = torch.cat(all_image_tensors, dim = 0)
            # print(input_ids.size())
            outputs = llava_generate(self.accelerator.unwrap_model(self.base), self.accelerator, 
            self.tokenizer, input_ids, all_image_tensors, self.config, image_sizes if self.use_anyres else None)
            # if self.eos_str is not None:
            #     output = output.split(self.eos_str)[0]
            # outputs.append(output)
            return outputs

    def get_log_prob(self, observation, actions):
        prompts = get_llava_prompts(observation)
        images = observation["image"]
        images = [get_image(image) if isinstance(image, str) else image for image in images]


        #modified action to ensure that the action ends with the eos token
        modified_action = []
        for i in range(len(actions)):
            if self.eos_str is not None and not self.eos_str in actions[i]:
                modified_action.append(actions[i] +self.eos_str)
            else:
                modified_action.append(actions[i])
        input_id_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX) for prompt in prompts]
        input_ids = torch.Tensor(pad_from_left(input_id_list, self.tokenizer.pad_token_id)).long().to(self.device)
        if self.use_anyres:
            image_tensors = process_images(images, self.image_processor, self.model_cfg).to(self.device, dtype=self.dtype)
        else:
            image_tensors = torch.cat([self.image_processor(image, return_tensors='pt')['pixel_values'].to(self.device, dtype=self.dtype) for image in images], dim = 0)
        image_sizes = [image.size for image in images]
        # print(image_tensors.shape)
        # image_tensors = torch.cat([self.image_processor(image, return_tensors='pt')['pixel_values'].to(self.device, dtype=self.accelerator.unwrap_model(self.base).dtype) for image in images], dim = 0)
        #remove the first special token
        output_ids = self.tokenizer(modified_action, return_tensors='pt', padding=True, max_length=self.max_new_tokens-1, truncation = True, add_special_tokens=True)["input_ids"].to(self.device)
        eos_ids = torch.tensor([self.tokenizer.eos_token_id], dtype=output_ids.dtype).broadcast_to(output_ids.shape[0],1).to(self.device)
        output_ids = torch.cat([output_ids, eos_ids], dim = 1)
        # print("evaluate_output_ids", output_ids)
        # if self.accelerator.is_main_process:
        #     print("output_ids", output_ids)
        return llava_evaluate(self.base, self.accelerator, input_ids, 
        output_ids, image_tensors, self.temperature, image_sizes if self.use_anyres else None).to(dtype=self.base.dtype)
