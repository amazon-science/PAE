import platform
import argparse
import random
import time
import json
import re
import os
import shutil
import logging
from PIL import Image
import boto3
import numpy as np
# import base64
import gym
from pae.misc import colorful_print


REDDIT = "http://WEBARENA_HOST:9999"
MAP = "http://WEBARENA_HOST:3000"
GITLAB = "http://WEBARENA_HOST:8023"
SHOPPING_ADMIN = "http://WEBARENA_HOST:7780/admin"
SHOPPING = "http://WEBARENA_HOST:7770"

from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

from .utils import get_web_element_rect, encode_image, extract_information,\
        clip_message_and_obs,get_pdf_retrieval_ans_from_claude

from .utils_eval import evaluate

import re
from .utils import replace_ec2_address

def get_image(image_path):
    with Image.open(image_path) as img:
        return np.array(img)


def driver_config(force_device_scale, headless, download_dir):
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")

    if force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )
    return options



def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(10)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    # outer_html = web_ele.get_attribute("outerHTML")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except Exception as e:
        # print(e)
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs


def exec_action_scroll(info, web_eles, driver_task, window_height):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-window_height*2//3});")
    else:
        if int(scroll_ele_number) <= len(web_eles):
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            raise NotImplementedError
            # element_box = obs_info[scroll_ele_number]['union_bound']
            # element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            # web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(10)
    

def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text):
    if it == 1:
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ]
        }
        init_msg_format['content'].append({'type': 'image', 'source': {
                                            'type': 'base64', 'media_type': 'image/png', 'data': web_img_b64}})
                                        #    "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image', 'source': {
                            'type': 'base64', 'media_type': 'image/png', 'data': web_img_b64}
                    }
                ]
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image', 'source': {
                            'type': 'base64', 'media_type': 'image/png', 'data': web_img_b64}
                    }
                ]
            }
        return curr_msg

def format_msg_with_act(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text, ac_tree):
    if it == 1:
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}\nYou can also look at the accessibility tree for more information.\n{ac_tree}"
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ]
        }
        init_msg_format['content'].append({'type': 'image', 'source': {
                                            'type': 'base64', 'media_type': 'image/png', 'data': web_img_b64}})
                                        #    "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}\nYou can also look at the accessibility tree for more information.\n{ac_tree}"},
                    {
                        'type': 'image', 'source': {
                            'type': 'base64', 'media_type': 'image/png', 'data': web_img_b64}
                    }
                ]
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}\nYou can also look at the accessibility tree for more information.\n{ac_tree}"},
                    {
                        'type': 'image', 'source': {
                            'type': 'base64', 'media_type': 'image/png', 'data': web_img_b64}
                    }
                ]
            }
        return curr_msg

class WebBroswerGym(gym.Env):
    def __init__(self,
                tasks,
                evaluator_prompt,
                max_attached_imgs = 3,
                evaluator_imgs = 3,
                max_iter = 15,
                download_dir = 'downloads', # download directory, need exist
                output_dir = 'results', # need exist
                fix_box_color = True,
                window_width = 1024,
                window_height = 768,
                force_device_scale = False,
                headless = True,
                use_webarena_eval=True,
                do_eval = True,
                verbose = False,
                webarena_host = 'WEBARENA_HOST',
                evaluator_model = None,
                region = 'us-west-2',
                aws_key_id = None,
                aws_secret_key = None,
                 ):
        # self.args = args
        self.tasks = tasks
        self.max_attached_imgs = max_attached_imgs
        self.fix_box_color = fix_box_color
        self.max_iter = max_iter
        self.download_dir_ini = download_dir
        self.ini_dir = output_dir
        self.window_width = window_width
        self.window_height = window_height
        self.force_device_scale = force_device_scale
        self.headless = headless
        self.use_webarena_eval = use_webarena_eval
        self.time_step = 0
        self.driver_task = None
        self.terminated = False
        self.verbose = verbose
        self.region = region
        self.evaluator_prompt = evaluator_prompt
        self.evaluator_imgs = evaluator_imgs
        self.evaluator_model = evaluator_model
        self.aws_key_id = aws_key_id
        self.aws_secret_key = aws_secret_key
        self.do_eval = do_eval

        # if not os.path.exists(os.path.join(self.ini_dir, "images")):
        #     os.makedirs(os.path.join(self.ini_dir, "images"))
        # self.img_path = str(os.path.join(self.ini_dir, "images"))
        self.webarena_host = webarena_host
        self.reddit = REDDIT.replace("WEBARENA_HOST", webarena_host)
        self.gitlab = GITLAB.replace("WEBARENA_HOST", webarena_host)
        self.shopping = SHOPPING.replace("WEBARENA_HOST", webarena_host)
        self.shopping_admin = SHOPPING_ADMIN.replace("WEBARENA_HOST", webarena_host)
        self.map = MAP.replace("WEBARENA_HOST", webarena_host)

    def step(self, action):
        try:
            # with timeout(4*60):
                return self._step(action)
        except Exception as e:
            if self.verbose:
                logging.error('Error when step the environment.')
                logging.error(e)
            self.close()
            return None

    def _step(self, action):
        if self.time_step > self.max_iter:
            self.terminated = True
        if self.terminated:
            return None
        self.time_step += 1
        # print(self.time_step)
        Terminated = False
        Reward = 0
        info = {}

         # remove the rects on the website
        if self.rects:
            # logging.info(f"Num of interactive elements: {len(self.rects)}")
            for rect_ele in self.rects:
                retry_time = 0
                for i in range(5):
                    try:
                        self.driver_task.execute_script("arguments[0].remove()", rect_ele)
                        break
                    except Exception as e:
                        if i >= 4:
                            if self.verbose:
                                print('error occured when remove rect')
                            self.terminated = True
                            return None
                        if self.verbose:
                            print(e)
                        time.sleep(2)
                        retry_time += 1
                        if retry_time > 10:
                            break


            self.rects = []
            # driver_task.save_screenshot(os.path.join(task_dir, 'screenshot{}_no_box.png'.format(it)))

        self.messages.append({'role': 'assistant', 'content': action})
        self.history.append(action+"\n")

        try:
            assert 'Thought:' in action and 'Action:' in action
        except AssertionError as e:
            if self.verbose:
                logging.error('Format ERROR: Both "Thought" and "Action" should be included in your reply.')
                logging.error(e)
            self.fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
            curr_msg = {
                'role': 'user',
                'content': self.fail_obs
            }
            self.messages.append(curr_msg)
            self.messages = clip_message_and_obs(self.messages, self.max_attached_imgs)
            self.history[-1] = self.history[-1] + "Observation: " + self.fail_obs
            # if self.use_claude_format:
            #     return self.messages, Reward, Terminated, info
            # else:
            #     return self.get_observation(), Reward, Terminated, info

            return self.get_observation(), Reward, Terminated, info

            # bot_thought = re.split(pattern, gpt_4v_res)[1].strip()
        chosen_action = re.split(self.pattern, action)[2].strip()

        action_key, info = extract_information(chosen_action)


        self.fail_obs = ""
        self.pdf_obs = ""
        self.warn_obs = ""
        # execute action
        try:
            window_handle_task = self.driver_task.current_window_handle
            self.driver_task.switch_to.window(window_handle_task)

            if action_key == 'click':
                click_ele_number = int(info[0])
                if click_ele_number >= len(self.web_eles):
                    raise NotImplementedError
                    # element_box = self.obs_info[click_ele_number]['union_bound']
                    # element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
                    # web_ele = self.driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
                else:
                    web_ele = self.web_eles[click_ele_number]
                
                ele_tag_name = web_ele.tag_name.lower()
                ele_type = web_ele.get_attribute("type")

                exec_action_click(info, web_ele, self.driver_task)

                # deal with PDF file
                current_files = sorted(os.listdir(self.download_dir))
                if current_files != self.download_files:
                    # wait for download finish
                    time.sleep(10)
                    current_files = sorted(os.listdir(self.download_dir))

                    current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in self.download_files and pdf_file.endswith('.pdf')]
                    if current_download_file:
                        print('start to solve pdf')
                        pdf_file = current_download_file[0]
                        pdf_file_path = os.path.join(self.download_dir, pdf_file)
                        if self.do_eval:
                            pdf_obs = get_pdf_retrieval_ans_from_claude(pdf_file_path, self.task['ques'], region_name=self.region,
                                                                    aws_key_id=self.aws_key_id, aws_secret_key=self.aws_secret_key)
                        else:
                            pdf_obs = ""
                        shutil.copy(pdf_file_path, self.task_dir)
                        self.pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                        print("pdf solved", pdf_obs)
                        # assert(False)
                    self.download_files = current_files

                if ele_tag_name == 'button' and ele_type == 'submit':
                    time.sleep(10)

            elif action_key == 'wait':
                time.sleep(5)

            elif action_key == 'type':
                type_ele_number = int(info['number'])
                if type_ele_number > len(self.web_eles):
                    raise NotImplementedError
                    # element_box = self.obs_info[type_ele_number]['union_bound']
                    # element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
                    # web_ele = self.driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
                else:
                    web_ele = self.web_eles[type_ele_number]
                
                self.warn_obs = exec_action_type(info, web_ele, self.driver_task)
                if 'wolfram' in self.task['web']:
                    time.sleep(5)

            elif action_key == 'scroll':
                exec_action_scroll(info, self.web_eles, self.driver_task, self.window_height)
                
            elif action_key == 'goback':
                self.driver_task.back()
                time.sleep(2)

            elif action_key == 'google':
                self.driver_task.get('https://www.google.com/')
                time.sleep(2)

            elif action_key == 'answer':
                if self.verbose:
                    logging.info(info['content'])
                    logging.info('finish!!')
                Terminated = True
                if self.do_eval:
                    Reward, eval_info = evaluate(self.messages, self.task_dir, self.img_path, self.task, self.driver_task, self.use_webarena_eval, 
                                             evaluator_prompt=self.evaluator_prompt,
                                             evaluator_imgs=self.evaluator_imgs,
                                             evaluator_model=self.evaluator_model,
                                             region=self.region,
                                             aws_key_id=self.aws_key_id,
                                             aws_secret_key=self.aws_secret_key)
                else:
                    eval_info = None
                    Reward = 0
                info['eval_info'] = eval_info
                info['reference'] = None
                if self.task.get('eval') is not None and self.task['eval'] is not None and self.task['eval'].get('reference_answer_raw_annotation') is not None:
                    info['reference'] = self.task['eval']['reference_answer_raw_annotation']
                # return None, 0, True, info

            else:
                raise NotImplementedError
            self.fail_obs = ""
        except Exception as e:
            if self.verbose:
                print('error occured while executing the action', e)
                logging.error('driver error info:')
                logging.error(str(action_key))
                logging.error(str(info))
            self.fail_obs = "The action you have chosen cannot be exected. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
            
            time.sleep(2)
        
        if self.time_step > self.max_iter:
            Terminated = True
        

        if Terminated:
            obs = self.get_observation()
            self.close()
            self.terminated = True
            # if self.use_claude_format:
            #     return self.messages, Reward, Terminated, info
            # else:
            #     return obs, Reward, Terminated, info
            return obs, Reward, Terminated, info

        try:
            alert = self.driver_task.switch_to.alert
            self.warn_obs = alert.text
            if self.verbose:
                logging.info(f"Alert detected with text: {alert.text}")
            alert.accept()
        except:
            pass
        
    
        if not self.fail_obs:
            try:
                self.rects, self.web_eles, self.web_eles_text, _ = get_web_element_rect(self.driver_task, fix_color=self.fix_box_color)
            except Exception as e:
                if self.verbose:
                    logging.error('Driver error when adding set-of-mark.')
                    logging.error(e)
                self.history[-1] = self.history[-1] + "Observation: The web page is not loaded properly. Please analyze the attached screenshot and give the Thought and Action. The screenshot is omitted here."

                
                self.close()
                print('web error occured, stop here')
                # if self.use_claude_format:
                #     return self.messages, Reward, Terminated, info
                # else:
                #     return self.get_observation(), Reward, Terminated, info
                return self.get_observation(), Reward, True, info

            img_path = os.path.join(self.task_dir, 'screenshot{}.png'.format(self.time_step))
            self.img_path = img_path
            self.driver_task.save_screenshot(img_path)

            # encode image
            # logging.info(img_path)
            b64_img = encode_image(img_path)

            curr_msg = format_msg(self.time_step, self.init_msg, self.pdf_obs, self.warn_obs, b64_img, self.web_eles_text)
            self.messages.append(curr_msg)
            if not self.pdf_obs:
                self.history[-1] = self.history[-1] + f"Observation:{self.warn_obs} Execution suc The screenshot is omitted here."
            else:
                self.history[-1] = self.history[-1] + f"Observation: {self.pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot is omitted here."
        else:
            curr_msg = {
                'role': 'user',
                'content': self.fail_obs
            }
            self.messages.append(curr_msg)
            self.history[-1] = self.history[-1] + self.fail_obs
        

        
        self.messages = clip_message_and_obs(self.messages, self.max_attached_imgs)

        # if self.use_claude_format:
        #     return self.messages, Reward, Terminated, info
        # else:
        #     return self.get_observation(), Reward, Terminated, info
        return self.get_observation(), Reward, Terminated, info


    def reset(self, task_id = None, task_new = None):
        try:
            # with timeout(4*60):
                self.terminated = False
                # Save Result file
                current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
                self.output_dir = os.path.join(self.ini_dir, current_time)
                os.makedirs(self.output_dir, exist_ok=True)
                self.download_dir = os.path.join(self.download_dir_ini, current_time)
                os.makedirs(self.download_dir, exist_ok=True)

                if task_id:
                    task = self.tasks[task_id % len(self.tasks)]
                elif task_new:
                    task = task_new
                else:
                    task = random.choice(self.tasks)
                self.task = task

                self.task_dir = os.path.join(self.output_dir, 'task{}'.format(task["id"]))
                if os.path.exists(self.task_dir):
                    shutil.rmtree(self.task_dir)
                os.mkdir(self.task_dir)
                # Create Driver
                options = driver_config(self.force_device_scale, self.headless, self.download_dir)
                if self.driver_task is not None:
                    self.driver_task.quit()
                    self.driver_task = None
                for i in range(3):
                    try:
                        self.driver_task = webdriver.Chrome(options=options)
                    except Exception as e:
                        if i >= 2:
                            if self.verbose:
                                logging.error('Driver error when creating the driver.')
                                logging.error(e)
                            self.close()
                            return None
                        time.sleep(5)

                # About window size, 765 tokens
                # You can resize to height = 512 by yourself (255 tokens, Maybe bad performance)
                self.driver_task.set_window_size(self.window_width, self.window_height)  # larger height may contain more web information
                self.pattern = r'Thought:|Action:|Observation:'

                # If it is webarena website, we need to login
                match self.task['web_name']:
                    case 'reddit':
                        username = "MarvelsGrantMan136"
                        password = "test1234"
                        for _ in range(3):
                            try:
                                self.driver_task.get(f"{self.reddit}/login")
                                # print(f"{self.reddit}/login")
                                # self.driver_task.save_screenshot('/home/ubuntu/test.jpg')
                                WebDriverWait(self.driver_task, 10).until(EC.presence_of_element_located((By.XPATH, "//label[text()='Username']/following-sibling::input")))
                                username_fiele = self.driver_task.find_element(By.XPATH, "//label[text()='Username']/following-sibling::input")
                                username_fiele.send_keys(username)
                                password_fiele = self.driver_task.find_element(By.XPATH, "//label[text()='Password']/following-sibling::input")
                                password_fiele.send_keys(password)
                                # self.driver_task.save_screenshot('/home/ubuntu/test2.jpg')
                                login_button = WebDriverWait(self.driver_task, 10).until(
                                    EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' and normalize-space(text())='Log in']")))
                                login_button.click()

                                time.sleep(10)
                                break
                            except Exception as e:
                                if _ >= 2:
                                    logging.error('error when login reddit.')
                                    logging.error(e)
                                    self.close()
                                    return None
                                time.sleep(5)
                    case 'gitlab':
                        username = "byteblaze"
                        password = "hello1234"

                        raise NotImplementedError

                    case 'shopping_admin':
                        username = "admin"
                        password = "admin1234"
                        for _ in range(3):
                            try:
                                self.driver_task.get(f"{self.shopping_admin}")
                                # print(f"{self.shopping_admin}")


                                username_field = self.driver_task.find_element(By.CSS_SELECTOR, "[placeholder='user name']")
                                username_field.send_keys(username) 

                                password_field = self.driver_task.find_element(By.CSS_SELECTOR, "[placeholder='password']")
                                password_field.send_keys(password)  

                                sign_in_button = WebDriverWait(self.driver_task, 10).until(
                                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.action-login.action-primary"))
                                )
                                sign_in_button.click()
                                # print('login into shopping_admin')
                                time.sleep(10)
                                break
                            except Exception as e:
                                if _ >= 2:
                                    logging.error('error when login shoping_admin.')
                                    logging.error(e)
                                time.sleep(5)

                    case 'shopping':
                        username = "emma.lopez@gmail.com"
                        password = "Password.123"

                        for _ in range(3):
                            try:

                                self.driver_task.get(f"{self.shopping}/customer/account/login/")
                                # print(f"{self.shopping}/customer/account/login/")




                                email_input = self.driver_task.find_element(By.ID, "email")
                                email_input.send_keys(username)

                                password_input = self.driver_task.find_element(By.ID, "pass")
                                password_input.send_keys(password)

                                sign_in_button = self.driver_task.find_element(By.ID, "send2")
                                sign_in_button.click()

                                time.sleep(10)
                                break
                            except Exception as e:
                                if _ >= 2:
                                    logging.error('error when login shopping.')
                                    logging.error(e)
                                time.sleep(5)
                retry_time = 0
                while True:
                    try:
                        self.driver_task.get(replace_ec2_address(self.task['web'], self.webarena_host))
                        break
                    except:
                        time.sleep(2)
                        retry_time += 1
                        if retry_time > 2:
                            if self.verbose:
                                logging.error(f'Driver error when loading the page. {self.task["web"]}')
                            self.terminated = True
                            return None
                try:
                    self.driver_task.find_element(By.TAG_NAME, 'body').click()
                except:
                    pass 
                # sometimes enter SPACE, the page will sroll down
                for i in range(3):
                    try:
                        self.driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
                        break
                    except Exception as e:
                        if i >= 2:
                            if self.verbose:
                                logging.error('Driver error when adding event listener.')
                                logging.error(e)
                            self.close()
                            return None
                        time.sleep(5)

                # We only deal with PDF file
                for filename in os.listdir(self.download_dir):
                    file_path = os.path.join(self.download_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                self.download_files = []  # sorted(os.listdir(args.download_dir))

                self.fail_obs = ""  # When error execute the action
                self.pdf_obs = ""  # When download PDF file
                self.warn_obs = ""  # Type warning
                # pattern = r'Thought:|Action:|Observation:'
                self.messages = []
                self.obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "

                self.init_msg = f"""Now given a task: {self.task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
                self.init_msg = self.init_msg.replace('https://www.example.com', self.task['web'])
                self.init_msg = self.init_msg + self.obs_prompt

                self.time_step = 1
                self.history = []


                info = None

                img_path = None

                try:
                    self.rects, self.web_eles, self.web_eles_text, _ = get_web_element_rect(self.driver_task, fix_color=self.fix_box_color)
                except Exception as e:
                    if self.verbose:
                        logging.error('Driver error when adding set-of-mark.')
                        logging.error(e)
                    self.close()
                    return None
                    # print('driver set up error!!!!!!!')
                    return self.get_observation(), info
                    # break

                img_path = os.path.join(self.task_dir, 'screenshot{}.png'.format(self.time_step))
                self.img_path = img_path
                self.driver_task.save_screenshot(img_path)

                # encode image
                # logging.info(img_path)
                b64_img = encode_image(img_path)

                
                    
                curr_msg = format_msg(self.time_step, self.init_msg, self.pdf_obs, self.warn_obs, b64_img, self.web_eles_text)
                # print(curr_msg)
                self.messages.append(curr_msg)
                    
                # print(self.messages)
                self.messages = clip_message_and_obs(self.messages, self.max_attached_imgs)

            
                # if self.use_claude_format:
                #     return self.messages, info
                # else:
                #     return self.get_observation(), info
                return self.get_observation(), info
        except Exception as e:
            logging.error('Error when reset the environment.')
            logging.error(e)
            self.close()
            return None
    
    def get_history(self):
        length = min(3, len(self.history))
        his = ""
        for i in range(-length, 0):
            his+= self.history[i] + "\n"
        return his

    
#     def get_observation(self):
#         observation = {
#             'image': self.img_path,
#             'task': self.init_msg,
#             'web_name': self.task['web_name'],
#             'button': self.web_eles_text,
#             'history': self.get_history(),
#             'message': self.messages,
#         }
#         return observation

    def get_observation(self):
        observation = {
            'image': self.img_path,
            'task': self.init_msg,
            'web_name': self.task['web_name'],
            'button': self.web_eles_text,
            'history': self.get_history(),
            'message': self.messages,
        }
        return observation

    def close(self):
        self.terminated = True
        try:
            shutil.rmtree(self.download_dir)
            # shutil.rmtree(self.output_dir)
            self.driver_task.quit()
        except:
            pass
        self.driver_task = None
        # shutil.rmtree(self.task_dir)
        # shutil.rmtree(self.output_dir)


        # files = os.listdir(self.output_dir)
        # for file in files:
        #     file_path = os.path.join(self.output_dir, file)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)
        #     elif os.path.isdir(file_path):
        #         shutil.rmtree(file_path)
        # shutil.rmtree(self.output_dir)

import concurrent
class BatchedWebEnv():
    def __init__(self,
                tasks,
                evaluator_prompt=None,
                batch_size = 1,
                max_attached_imgs = 3,
                evaluator_imgs = 3,
                max_iter = 15,
                download_dir = 'downloads', # download directory, need exist
                output_dir = 'results', # need exist
                fix_box_color = True,
                window_width = 1024,
                window_height = 768,
                force_device_scale = True,
                headless = True,
                use_webarena_eval=True,
                do_eval = True,
                random_task = False,
                verbose = False,
                webarena_host = "ec2-18-236-182-17.us-west-2.compute.amazonaws.com",
                evaluator_model = "anthropic.claude-3-sonnet-20240229-v1:0",
                region = 'us-west-2',
                ssh_key_path = "/home/ubuntu/.ssh/id_rsa",
                aws_key_id = None,
                aws_secret_key = None
                 ):
        if do_eval:
            colorful_print("Warning: The evaluation is enabled, please make sure you have the correct AWS credentials and the correct region.")
        else:
            colorful_print("Warning: The evaluation is disabled, the evaluation will not be performed.")
        self.tasks = tasks
        self.max_attached_imgs = max_attached_imgs
        self.fix_box_color = fix_box_color
        self.max_iter = max_iter
        self.download_dir = download_dir
        self.output_dir = output_dir
        self.window_width = window_width
        self.window_height = window_height
        self.force_device_scale = force_device_scale
        self.headless = headless
        self.use_webarena_eval = use_webarena_eval
        self.webarena_host = webarena_host
        self.ssh_key_path = ssh_key_path

        self.random_task = random_task

        self.batch_size = batch_size
        self.envs = []
        for i in range(self.batch_size):
            os.makedirs(os.path.join(self.output_dir, f'batch{i}'), exist_ok=True)
            os.makedirs(os.path.join(self.download_dir, f'batch{i}'), exist_ok=True)
            env = WebBroswerGym(tasks,evaluator_prompt, 
                                max_attached_imgs, 
                                evaluator_imgs,
                                max_iter, 
                                os.path.join(self.download_dir, f'batch{i}'), 
                                os.path.join(self.output_dir, f'batch{i}'),
                                fix_box_color, 
                                window_width, 
                                window_height, 
                                force_device_scale, 
                                headless, 
                                use_webarena_eval, 
                                do_eval=do_eval,
                                verbose=verbose, 
                                webarena_host=webarena_host, 
                                evaluator_model=evaluator_model,
                                region=region,
                                aws_key_id=aws_key_id,
                                aws_secret_key=aws_secret_key)
            self.envs.append(env)
        
    def reset_server(self):
        import paramiko 
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh_key_path = self.ssh_key_path
        key = paramiko.RSAKey.from_private_key_file(ssh_key_path)
        ssh.connect(
            hostname=self.webarena_host,
            username="ubuntu",
            pkey=key
        )
        commands = [
            "bash ~/reset.sh",
            "bash ~/setup.sh",
            "bash ~/setup.sh"
        ]
        for command in commands:
            print(f"Executing: {command}")
            stdin, stdout, stderr = ssh.exec_command(command)
            
            while not stdout.channel.exit_status_ready():
                if stdout.channel.recv_ready():
                    output = stdout.read().decode()
                    print(output, end="")
                time.sleep(1)  

            output = stdout.read().decode()
            error = stderr.read().decode()

            print("Output:")
            print(output)

            if error:
                print("Error:")
                print(error)

        ssh.close()

    def reset(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if self.random_task:
                jobs = [executor.submit(env.reset) for env in self.envs]
            else:
                jobs = [executor.submit(env.reset, i) for i, env in enumerate(self.envs)]
            observations = [job.result() for job in jobs]
        return observations
    
    def step(self, actions):
        assert len(actions) == self.batch_size
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env.step, action) for env, action in zip(self.envs, actions)]
            results = [job.result() for job in jobs]

        return results
