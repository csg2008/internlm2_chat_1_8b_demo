import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = '/home/xlab-app-center/internlm2-chat-1-8b'

if not os.path.exists(base_path):
    os.system(f'git clone https://code.openxlab.org.cn/csg2008/internlm2_chat_1_8b_demo.git {base_path}')
    os.system(f'cd {base_path} && git lfs pull')

title = '基于书生浦语1.8B的自我认知实验'
description = '微调 internlm2-chat-1-8b 模型调整自我认识的实验，课程笔记: https://github.com/csg2008/InternLMAgricultureAssistant/blob/main/note/lesson4.md'
tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response, history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat, title=title, description=description).queue(1).launch()
