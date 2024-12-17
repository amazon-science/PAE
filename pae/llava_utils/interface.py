import torch
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import LlamaForCausalLM

def find_all_linear_names(model,train_vision=None):
    cls = torch.nn.Linear
    lora_module_names = set()
    if train_vision == 'all':
        multimodal_keywords = ['vision_resampler']
    elif train_vision == 'projector':
        multimodal_keywords = ['vision_resampler', 'vision_tower']
    else:
        multimodal_keywords = ['vision_resampler', 'vision_tower', 'mm_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if "value_head" in name:
            # value head should not be part of the adapter
            continue
        if isinstance(module, cls):
            names = name.split('.')

            if "0"<=names[-1] and names[-1]<="9":
                lora_module_names.add(names[0] if len(names) == 1 else names[-2]+"."+names[-1])
            else:
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    # lora_module_names = ['k_proj', 'v_proj', 'q_proj'] 
    print(list(lora_module_names)) 
    return list(lora_module_names)

"""
The format of the observation is a dictionary with the following keys:
- task: a list of strings or a string, each string is a task description
- button: a list of strings or a string, each string is a button description+
"""
def get_llava_prompts(observation):
    tasks = observation["task"]
    buttons = observation["button"]
    if not isinstance(tasks, list):
        tasks = [tasks]
    if not isinstance(buttons, list):
        buttons = [buttons]
    # print(observation["history"])
    assert 'history' in observation, 'history key not found in observation dictionary. Please provide a history key in the observation dictionary.'
    if "history" in observation:
        histories = observation["history"]
        if not isinstance(histories, list):
            histories = [histories]
    else:
        histories = [""]*len(tasks)
    prompts = []
    for task, button, history in zip(tasks, buttons, histories):
        system_message = "You are a webagent, your need to output thoughts and actions given a webpage screenshot and a task to complete."
        system_message += f"The history of the interaction is: {history[-512:]}"
        #only use the first 1024 characters of the button and task
        qs = DEFAULT_IMAGE_TOKEN + "\n" + system_message + f"Your current task is: {task[:512]}" + f"Your current buttons are: {button[:512]}"
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
    # print(prompts)
    return prompts

def llava_generate(base, accelerator, tokenizer, input_ids, image_tensor, args, image_sizes=None):
    # image_tensor = image_tensor.to(base.device, dtype = accelerator.unwrap_model(base).dtype)
    # all_inputs_embeds = []
    # for input_id, image, image_size in zip(input_ids, image_tensor, image_sizes):
    #     _, _, _, _, inputs_embeds, _ = accelerator.unwrap_model(base).prepare_inputs_labels_for_multimodal(input_id.unsqueeze(0), None, None, None, None, [image], [image_size])
    #     all_inputs_embeds.append(inputs_embeds)
    # inputs_embeds = torch.cat(all_inputs_embeds, dim = 0)
    # _, _, _, _, inputs_embeds, _ = accelerator.unwrap_model(base).prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, image_tensor, image_sizes)
    with torch.inference_mode():
        outputs = accelerator.unwrap_model(base).generate(input_ids, image_tensor, image_sizes,
                                                            do_sample=True,
                                                            temperature=args.temperature,
                                                            num_beams=args.num_beams,
                                                            max_new_tokens=args.max_new_tokens,
                                                            use_cache=True,
                                                            output_scores=True,
                                                            output_hidden_states=True,
                                                            return_dict_in_generate=True,
                                                            pad_token_id=tokenizer.eos_token_id,)
        # outputs = LlamaForCausalLM.generate(
        # accelerator.unwrap_model(base),
        # inputs_embeds = inputs_embeds,
        # # inputs = input_ids,
        # # images = image_tensor,
        # do_sample=True,
        # temperature=args.temperature,
        # num_beams=args.num_beams,
        # max_new_tokens=args.max_new_tokens,
        # use_cache=True,
        # output_scores=True,
        # output_hidden_states=True,
        # return_dict_in_generate=True,
        # pad_token_id=tokenizer.eos_token_id,)
        if isinstance(outputs, dict):
            output_ids = outputs['sequences']
        else:
            output_ids = outputs
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs

# def llava_generate(base, accelerator, tokenizer, input_ids, image_tensor, args):
#     # image_tensor = image_tensor.to(base.device, dtype = accelerator.unwrap_model(base).dtype)
#     with torch.inference_mode():
#         outputs = base.generate(
#         inputs = input_ids,
#         images = image_tensor,
#         do_sample=True,
#         temperature=args.temperature,
#         num_beams=args.num_beams,
#         max_new_tokens=args.max_new_tokens,
#         use_cache=True,
#         output_scores=True,
#         output_hidden_states=True,
#         return_dict_in_generate=True,
#         pad_token_id=tokenizer.eos_token_id,)
#         output_ids = outputs['sequences']
#     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     return outputs
    # padded_output_ids = torch.zeros(output_ids.size(0), 2*args.max_new_tokens).to(dtype=output_ids.dtype, device = output_ids.device)
    # padded_output_ids[:, :output_ids.size(1)] = output_ids
    # with torch.no_grad():
    #     values, sum_log_probs, action_tokens_log_prob = llava_evaluate(value_model, input_ids, padded_output_ids, image_tensor, args.temperature, args.thought_prob_coef)
    # return values, padded_output_ids, outputs, sum_log_probs, action_tokens_log_prob

def llava_evaluate(base, accelerator, input_ids, output_ids, image_tensor, temperature, image_sizes):
    if output_ids.size(0) != 1:
        input_ids = input_ids.broadcast_to(output_ids.size(0), input_ids.size(-1))
    image_tensor = image_tensor.to(base.device, dtype=accelerator.unwrap_model(base).dtype)
    output_ids = output_ids.to(base.device)
    input_ids = input_ids.to(base.device)
    # print(torch.cat([input_ids, output_ids], dim = 1).shape)
    # _, _, _, _, inputs_embeds, _ = accelerator.unwrap_model(base).prepare_inputs_labels_for_multimodal(torch.cat([input_ids, output_ids], dim = 1), None, None, None, None, image_tensor, image_sizes)
    # #calling the model
    # inputs_embeds = inputs_embeds.to(base.device, dtype = accelerator.unwrap_model(base).dtype)
    # #omit the first output token
    # outputs = base(
    #     inputs_embeds = inputs_embeds,
    #     output_hidden_states = True,
    #     )
    inputs_embeds, outputs = base(torch.cat([input_ids, output_ids], dim = 1), image_tensor, image_sizes = image_sizes)
    scores = outputs.logits

    input_token_len = inputs_embeds.shape[1] - output_ids.shape[1]
    hidden_states = outputs.hidden_states[-1][:, input_token_len-1]
    scores = scores * (1/temperature)
    scores = scores
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs
    output_ids_mask = (output_ids != 0)[:, 1:]
    selected_log_probs = output_ids_mask*torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
    return selected_log_probs.mean(dim = 1)



    # unfolded = output_ids.unfold(dimension=-1, size=3, step=1)
    # target = torch.tensor([29908,2467,1115]).to(base.device)
    # # tokens for text string:'"action":' (torch.tensor([[29908,2467,1115]]))
    # matches = (unfolded == target).all(dim = -1)
    # match_index = matches.nonzero(as_tuple=True)[-1]
    # if match_index.shape[0] > 1:
    #     match_index = match_index[-1].unsqueeze(0)
    # else:
    #     try:
    #         match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
    #     except:
    #         sum_log_prob = torch.tensor([-2]).to(base.device)
    #         action_tokens_log_prob = torch.tensor([-1]).to(base.device)
    #         return values, sum_log_prob, action_tokens_log_prob
    # ## omitting the second token for calculating log prob, because its logprb is very very small
    # thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)
    # action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
    # sum_log_prob = thought_prob_coef*thought_log_prob + action_tokens_log_prob
    # return values, sum_log_prob, action_tokens_log_prob



SYSTEM_PROMPT = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will feature Numerical Labels placed in the TOP LEFT corner of each Web Element.
Carefully analyze the visual information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions:
1. Click a Web Element.
2. Delete existing content in a textbox and then type content. 
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
5. Go back, returning to the previous webpage.
6. Google, directly jump to the Google search page. When you can't find information in some websites, try starting over with Google.
7. Answer. This action should only be chosen when all questions in the task have been solved.

Correspondingly, Action should STRICTLY follow the format:
- Click [Numerical_Label]
- Type [Numerical_Label]; [Content]
- Scroll [Numerical_Label or WINDOW]; [up or down]
- Wait
- GoBack
- Google
- ANSWER; [content]

Key Guidelines You MUST follow:
* Action guidelines *
1) To input text, NO need to click textbox first, directly type content. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
2) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
3) Execute only one action per iteration. 
4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
5) When a complex Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the task when ANSWER. 
* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
3) Focus on the numerical labels in the TOP LEFT corner of each rectangle (element). Ensure you don't mix them up with other numbers (e.g. Calendar) on the page.
4) Focus on the date in task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
5) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

Your reply should strictly follow the format:
Thought: {Your brief thoughts (briefly summarize the info that will help ANSWER)}
Action: {One Action format you choose}

Then the User will provide:
Observation: {A labeled screenshot Given by User}"""


"""
The format of the observation is a dictionary with the following keys:
- task: a list of strings or a string, each string is a task description
- button: a list of strings or a string, each string is a button description+
"""
def get_zeroshot_prompts(observation):
    tasks = observation["task"]
    buttons = observation["button"]
    if not isinstance(tasks, list):
        tasks = [tasks]
    if not isinstance(buttons, list):
        buttons = [buttons]
    # print(observation["history"])
    assert 'history' in observation, 'history key not found in observation dictionary. Please provide a history key in the observation dictionary.'
    if "history" in observation:
        histories = observation["history"]
        if not isinstance(histories, list):
            histories = [histories]
    else:
        histories = [""]*len(tasks)
    prompts = []
    for task, button, history in zip(tasks, buttons, histories):
        system_message = SYSTEM_PROMPT
        system_message += f"The history of the interaction is: {history[-512:]}"
        #only use the first 1024 characters of the button and task
        qs = DEFAULT_IMAGE_TOKEN + "\n" + system_message + f"Your current task is: {task[:512]}" + f"Your current buttons are: {button[:512]}"
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
    # print(prompts)
    return prompts