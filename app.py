import os

import gradio as gr
from lagent.llms import HFTransformer
from lagent.llms.meta_template import INTERNLM2_META
from lagent.agents.internlm2_agent import INTERPRETER_CN, META_CN, PLUGIN_CN, Internlm2Agent, Internlm2Protocol
from lagent.schema import AgentStatusCode

print("gradio version: ", gr.__version__)

local_model_path = '/root/ft/final_model'
if os.path.exists(local_model_path):
    base_path = local_model_path
else:
    download_model_path = '/home/xlab-app-center/internlm2-chat-1-8b'
    remote_repo_url = 'https://code.openxlab.org.cn/csg2008/internlm2_chat_1_8b_demo.git'
    if not os.path.exists(download_model_path):
        os.system(f'git clone {remote_repo_url} {download_model_path}')
        os.system(f'cd {download_model_path} && git lfs pull')

    base_path = download_model_path

title = '基于书生浦语1.8B的自我认知实验'
description = '微调 internlm2-chat-1-8b 模型调整自我认识的实验，课程笔记: https://github.com/csg2008/InternLMAgricultureAssistant/blob/main/note/lesson4.md'

model = HFTransformer(
    path=base_path,
    meta_template=INTERNLM2_META,
    max_new_tokens=1024,
    top_p=0.8,
    top_k=None,
    temperature=0.1,
    repetition_penalty=1.0,
    stop_words=['<|im_end|>'])


chatbot = Internlm2Agent(
    llm=model,
    plugin_executor=None,
    interpreter_executor=None,
    protocol=Internlm2Protocol(
        meta_prompt=META_CN,
        interpreter_prompt=INTERPRETER_CN,
        plugin_prompt=PLUGIN_CN,
        tool=dict(
            begin='{start_token}{name}\n',
            start_token='<|action_start|>',
            name_map=dict(
                plugin='<|plugin|>', interpreter='<|interpreter|>'),
            belong='assistant',
            end='<|action_end|>\n',
        ),
    ),
)

def chat (message, history):
    # 转换历史消息格式及添加最新消息
    prompts = []
    for user, assistant in history:
        prompts.append(
            {
                'role': 'user',
                'content': user
            }
        )
        prompts.append(
            {
                'role': 'assistant',
                'content': assistant
            })
    prompts.append(
        {
            'role': 'user',
            'content': message
        }
    )

    response = ''
    for agent_return in chatbot.stream_chat(prompts):
        status = agent_return.state
        if status not in [
                AgentStatusCode.STREAM_ING, AgentStatusCode.CODING,
                AgentStatusCode.PLUGIN_START
        ]:
            continue

        if isinstance(agent_return.response, dict):
            action = f"\n\n {agent_return.response['name']}: \n\n"
            action_input = agent_return.response['parameters']
            if agent_return.response['name'] == 'IPythonInterpreter':
                action_input = action_input['command']
            response = action + action_input
        else:
            response = agent_return.response

        yield response

    history.append([message, response])

gr.ChatInterface(chat, title=title, description=description).queue(1).launch()
