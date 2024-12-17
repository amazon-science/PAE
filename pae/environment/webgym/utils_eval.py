import argparse
import os
import json
import time
import re
import base64
from typing import Any

from nltk.tokenize import word_tokenize

# from openai import OpenAI
import boto3

import collections
import urllib
import html



USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """

USER_REFERENCE_PROMPT = """TASK: <task>
Result Response: <answer>
Reference Response: <reference>
<num> screenshots at the end: """


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# WebArena evaluator functions
def llm_fuzzy_match(pred: str, reference: str, question: str, claude_client) -> float:
    """Check whether the prediction matches the reference with Claude3"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by correct/incorrect/partially correct."
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': message}
            ]
        }
    ]

    for i in range(3):
        try:
            # print('Calling Claude3 API to get the auto evaluation......')
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages,
                "temperature": 0,
                "top_p": 0.7,
                'system' : "You are a helpful assistant"
            }
            claude_response = claude_client.invoke_model(
                modelId = "anthropic.claude-3-sonnet-20240229-v1:0",
                body=json.dumps(request_body)
            )
            result = json.loads(claude_response.get("body").read())
            break
        except Exception as e:
            print(e)
            print(e.__ne__)
            if i == 2:
                return 0
            if e == 'ModelErrorException':
                time.sleep(10)
            elif type(e).__ne__ == 'APIError':
                time.sleep(15)
            elif type(e).__ne__ == 'InvalidRequestError':
                exit(0)
            else:
                time.sleep(10)
    response = result['content'][0]['text'].lower()

    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response
        return 1.0


def llm_ua_match(pred: str, reference: str, question: str, claude_client) -> float:
    """Check whether the prediction matches the reference with Claude"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': message}
            ]
        }
    ]

    while True:
        try:
            # print('Calling Claude3 API to get the auto evaluation......')
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages,
                "temperature": 0,
                "top_p": 0.7,
                'system' : "You are a helpful assistant"
            }
            claude_response = claude_client.invoke_model(
                modelId = "anthropic.claude-3-sonnet-20240229-v1:0",
                body=json.dumps(request_body)
            )
            result = json.loads(claude_response.get("body").read())
            break
        except Exception as e:
            print(e)
            print(e.__ne__)
            if e == 'ModelErrorException':
                time.sleep(10)
            elif type(e).__ne__ == 'APIError':
                time.sleep(15)
            elif type(e).__ne__ == 'InvalidRequestError':
                exit(0)
            else:
                time.sleep(10)
    response = result['content'][0]['text'].lower()
    
    
    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0


class StringEvaluator():
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    def __init__(self, client):
        self.client = client

    def clean_answer(self, answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    def exact_match(self, ref: str, pred: str) -> float:
        return float(
            self.clean_answer(pred)
            == self.clean_answer(ref)
        )

    def must_include(self, ref: str, pred: str, tokenize: bool = False) -> float:
        clean_ref = self.clean_answer(ref)
        clean_pred = self.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0)
        if (
            tokenize
            and len(clean_ref) == 1
            and len(word_tokenize(clean_ref)) == 1
        ):
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)

    def fuzzy_match(self, ref: str, pred: str, intent: str, client) -> float:
        return llm_fuzzy_match(pred, ref, intent, client)

    def ua_match(self, ref: str, pred: str, intent: str, client) -> float:
        return llm_ua_match(pred, ref, intent, client)

    def __call__(
        self,
        messages,
        eval,
        driver
    ) -> float:
        task_info = messages[0]["content"]
        if type(task_info) == list:
            task_info = task_info[0]["text"]


        pattern = r"Now given a task:(.+?)Please interact with"
        matches = re.search(pattern, task_info)
        task_content = matches.group(1).strip()

        ans_info = messages[-1]["content"]
        if 'Action: ANSWER' not in ans_info:
            # print('Not find answer for ')
            # print()
            # print('reward: 0')
            return 0
        pattern_ans = r"ANSWER[; ]+\[?(.[^\]]*)\]?"
        matches_ans = re.search(pattern_ans, ans_info)
        pred = matches_ans.group(1).strip()


        score = 1.0
        for approach, value in eval["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(value, pred)

                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                case "fuzzy_match":
                    intent = task_content
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        score *= self.exact_match(ref=value, pred=pred)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if score != 1:
                            score = 1.0 * self.ua_match(
                                intent=task_content,
                                ref=eval["string_note"],
                                pred=pred,
                                client=self.client,
                            )
                    else:
                        assert isinstance(value, list)
                        for reference in value:
                            score *= self.fuzzy_match(
                                ref=reference, pred=pred, intent=intent, client=self.client,
                            )
        return score


class URLEvaluator():
    """Check URL matching"""

    
    def __call__(
            self,
            messages,
            eval,
            driver
            ):
        def clean_url(url: str) -> str:
            url = str(url)
            url = url.rstrip("/")
            return url

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            """Parse a URL into its base, path, and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = parsed_url.netloc + parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            return base_path, query

        def parse_urls(
            urls: list[str],
        ) -> tuple[list[str], dict[str, set[str]]]:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries

        pred = clean_url(driver.current_url)
        ref_urls = eval["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        matching_rule = eval.get("url_note", "GOLD in PRED")

        # print(f"pred: {pred}")
        # print(f"ref_urls: {ref_urls}")

        if matching_rule == "GOLD in PRED":
            ref_base_paths, ref_queries = parse_urls(ref_urls)
            pred_base_paths, pred_query = parse_url(pred)

            # print(f"ref_base_paths: {ref_base_paths}")
            # print(f"ref_queries: {ref_queries}")
            # print(f"pred_base_paths: {pred_base_paths}")
            # print(f"pred_query: {pred_query}")

            base_score = float(
                any(
                    [
                        ref_base_path in pred_base_paths
                        for ref_base_path in ref_base_paths
                    ]
                )
            )
            query_score = 1.0
            for k, possible_values in ref_queries.items():
                query_score *= float(
                    any(
                        possible_ref_value in pred_query.get(k, [])
                        for possible_ref_value in possible_values
                    )
                )
            # print(f"base_score: {base_score}", f"query_score: {query_score}")
            score = base_score * query_score
        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")
        return score

class HTMLContentEvaluator():
    """Check whether the contents appear in the page"""

    def clean_answer(self, answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    def exact_match(self, ref: str, pred: str) -> float:
        return float(
            self.clean_answer(pred)
            == self.clean_answer(ref)
        )

    def must_include(self, ref: str, pred: str, tokenize: bool = False) -> float:
            clean_ref = self.clean_answer(ref)
            clean_pred = self.clean_answer(pred)
            # tokenize the answer if the ref is a single word
            # prevent false positive (e.g, 0)
            if (
                tokenize
                and len(clean_ref) == 1
                and len(word_tokenize(clean_ref)) == 1
            ):
                tok_pred = word_tokenize(clean_pred)
                return float(clean_ref in tok_pred)
            else:
                return float(clean_ref in clean_pred)

    def __call__(
            self,
            messages,
            eval,
            driver
            ):
        targets = eval["program_html"]
        score = 1.0
        for target in targets:
            target_url = target["url"]
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", driver.current_url)
                target_url = eval(func)
            
            locator = target["locator"]

            if target_url != "last":
                driver.get(target_url)
                time.sleep(3)

            # print(f"locator: {locator}")
            
            if not locator.strip():
                selected_element = driver.page_source
            elif locator.startswith("document.") or locator.startswith("[...document."):
                if "prep_actions" in target:
                    try:
                        for prep_action in target["prep_actions"]:
                            driver.execute_script(f"return {prep_action}")
                    except Exception:
                        pass
                try:
                    # print('here')
                    # print(f"return {locator}")
                    selected_element = str(driver.execute_script(f"return {locator}"))
                    # print('real selected element', selected_element)
                    if not selected_element:
                        selected_element = ""
                except Exception:
                    selected_element = ""
            elif locator.startswith("func:"):
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                selected_element = eval(func)
            else:
                raise ValueError(f"Unknown locator: {locator}")
            
            # print(f"selected_element: {selected_element}")
            
            selected_element = html.unescape(selected_element)

            if "exact_match" in target["required_contents"]:
                required_contents = target["required_contents"]["exact_match"]
                cur_score = self.exact_match(ref=required_contents, pred=selected_element)
                score *= float(cur_score)
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    content_or = content.split(" |OR| ")
                    cur_score = any([self.must_include(ref=content, pred=selected_element, tokenize=False) for content in content_or])
                    score *= float(cur_score)
            else:
                raise ValueError(f"Unknown required_contents: {target['required_contents'].keys()}")
        return score




def webarena_eval(messages, eval, driver, client):
    # print('begin webarena match')

    eval_types = eval["eval_types"]
    evaluators = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator(client))
            case "url_match":
                evaluators.append(URLEvaluator())
            case "program_html":
                evaluators.append(HTMLContentEvaluator())
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")
    

    score = 1.0
    for evaluator in evaluators:
        cur_score = evaluator(messages, eval, driver)
        score *= cur_score

    # print('reward:', score)

    return score, None



# Autonomous evaluation using Claude3
def auto_eval_by_claude3(it_messages, process_dir, img_path, claude_client, api_model, img_num, task, evaluator_prompt):

    reference = None
    if task.get('eval') is not None and task['eval'] is not None and task['eval'].get('reference_answer_raw_annotation') is not None:
        reference = task['eval']['reference_answer_raw_annotation']
    
    if len(it_messages) == 0:
        return None

    task_info = it_messages[0]["content"]
    if type(task_info) == list:
        task_info = task_info[0]["text"]
    # print(task_info)
    assert 'Now given a task' in task_info
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info)
    task_content = matches.group(1).strip()

    ans_info = it_messages[-1]["content"]
    if 'Action: ANSWER' not in ans_info:
        # print('Not find answer for ')
        # print()
        # print('reward: 0')
        return 0
    pattern_ans = r"ANSWER[; ]+\[?(.[^\]]*)\]?"
    matches_ans = re.search(pattern_ans, ans_info)
    answer_content = matches_ans.group(1).strip()

    screenshots = [int(f[10:].split('.png')[0]) for f in os.listdir(process_dir) if '.png' in f]
    screenshots.sort()
    screenshots = screenshots[-img_num:]

    whole_content_img = []
    for screenshot_id in screenshots:
        img_path = os.path.join(process_dir, f'screenshot{screenshot_id}.png')
        b64_img = encode_image(img_path)
        whole_content_img.append(
            {
                'type': 'image',
                'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64_img}
            }
        )



    user_prompt_tmp = USER_PROMPT.replace('<task>', task_content)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(img_num))
    messages = [
        # {'role': 'system', 'content': SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_prompt_tmp}
            ]
            + whole_content_img
            + [{'type': 'text', 'text': "Your verdict:\n"}]
        }
    ]
    # print(messages)
    for i in range(3):
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages,
                "temperature": 0,
                "top_p": 0.7,
                'system' : evaluator_prompt,
            }
            claude_response = claude_client.invoke_model(
                modelId = api_model,
                body=json.dumps(request_body)
            )
            result = json.loads(claude_response.get("body").read())
            break
        except Exception as e:
            print(e)
            print(e.__ne__)
            if i == 2:
                return 0
            if e == 'ModelErrorException':
                time.sleep(10)
            elif type(e).__ne__ == 'APIError':
                time.sleep(15)
            elif type(e).__ne__ == 'InvalidRequestError':
                exit(0)
            else:
                time.sleep(10)
    claude_3_res = result['content'][0]['text']
    # gpt_4v_res = openai_response.choices[0].message.content
    print_message = messages[0]
    for idx in range(len(print_message['content'])):
        if print_message['content'][idx]['type'] == 'image':
            print_message['content'][idx]['source'] = {"url": "data:image/png;base64, b64_img"}


    auto_eval_res = 0 if 'NOT SUCCESS' in claude_3_res else 1
    if 'SUCCESS' not in claude_3_res:
        auto_eval_res = 0
    return auto_eval_res, claude_3_res


def evaluate(messages, process_dir, img_path, task, driver, use_webarena_evaluator=True, evaluator_model = "anthropic.claude-3-sonnet-20240229-v1:0",
             region="us-west-2", evaluator_prompt = None, evaluator_imgs = 3, aws_key_id = None, aws_secret_key = None):
    if aws_key_id is not None and aws_secret_key is not None:
        client = boto3.client(service_name="bedrock-runtime", region_name=region, aws_access_key_id=aws_key_id, aws_secret_key=aws_secret_key)
    else:
        client = boto3.client(service_name="bedrock-runtime", region_name=region)
    if task.get('eval') is not None and task['eval'] is not None and use_webarena_evaluator:
        return webarena_eval(messages, task['eval'], driver, client)
    else:
        return auto_eval_by_claude3(messages, process_dir, img_path, client, 
                                    api_model=evaluator_model, 
                                    img_num=evaluator_imgs, 
                                    task=task, 
                                    evaluator_prompt = evaluator_prompt)
