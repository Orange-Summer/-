import pprint
from qwen_agent.agents import Assistant


def init_agent_service():
    # 步骤 2：配置您所使用的 LLM。
    llm_cfg = {
        'model': 'Qwen1.5-14B',
        'model_server': 'http://10.58.0.2:8000/v1',
        'api_key': 'None',
    }

    # 步骤 3：创建一个智能体。
    system_instruction = '''你是一个选课系统。
    初始用户没有选择上的课程
    根据用户的请求实现以下功能:
    - 查询：带有筛选的查询，可以筛选必修或选修。根据描述返回用户最为感兴趣的课程，例如用户喜欢体育，则将羽毛球等放在前面。
    - 选课：选择需要的课程，智能返回结果
        成功返回选课结果
        未成功返回错误
    - 删除：删除选择的课程，智能返回结果。
    - 当用户在选课和删除时提供的课程不准确时，智能提供可能用户想提的课程。
    你总是用中文回复用户。'''
    files = ['./courses.json']  # 给智能体一个 json 文件阅读。
    bot = Assistant(llm=llm_cfg,
                    system_message=system_instruction,
                    files=files)
    return bot


def app_tui():
    bot = init_agent_service()

    # 步骤 4：作为聊天机器人运行智能体。
    messages = []  # 这里储存聊天历史。
    while True:
        query = input('用户请求: ')
        # 将用户请求添加到聊天历史。
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run(messages):
            # 流式输出。
            print('机器人回应:')
            pprint.pprint(response, indent=2)
        # 将机器人的回应添加到聊天历史。
        messages.extend(response)


if __name__ == '__main__':
    app_tui()
