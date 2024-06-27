import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI

# 设置API环境变量
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"

# 初始化大模型
llm_chat = ChatOpenAI(model_name="Qwen1.5-14B")

# 初始化嵌入模型
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# 连接 Milvus 向量数据库
db = Milvus(embedding_function=embedding, collection_name="arXiv",
            connection_args={"host": "172.29.4.47", "port": "19530"})


# 使用大模型对用户输入的问题进行润色
def refine_question(question):
    prompt = f"请将以下问题翻译成英文并改写得更适合学术搜索：{question}"
    refined_question = llm_chat.invoke(prompt)
    return refined_question.content.strip()


# 获取与问题相关的论文 abstract
def get_relevant_papers(question, top_k=5, iterations=3):
    for _ in range(iterations):
        question_embedding = embedding.embed_query(question)
        search_results = db.similarity_search_by_vector(question_embedding, k=top_k)

        # 判断结果是否满意，可以根据某些条件进行判断
        if search_results and is_results_satisfactory(search_results):
            return search_results

        # 如果不满意，可以再次润色问题
        question = refine_question(question)

    return search_results


# 可以根据具体的业务需求定义满意的标准，例如结果的相关性评分等
def is_results_satisfactory(results):
    return len(results) > 0


# 使用大模型生成答案
def answer_question(question, search_results):
    abstracts = [result.page_content for result in search_results]
    sources = [result.metadata['title'].replace("\n", "").replace("  ", " ") for result in search_results]

    context = " ".join(abstracts)
    prompt = f"根据以下内容回答问题：\n{context}\n问题：{question}\n答案："

    answer = llm_chat.invoke(prompt)
    return answer.content.strip(), sources


def main():
    while True:
        user_question = input("请输入您的问题：")

        # 对问题进行润色
        refined_question = refine_question(user_question)

        # 获取相关论文摘要
        search_results = get_relevant_papers(refined_question)

        # 生成答案
        answer, sources = answer_question(user_question, search_results)

        # 输出答案和来源
        print(f"答案：{answer}")
        print("信息来源：")
        for source in sources:
            print(source)

        # 判断是否继续
        cont = input("是否继续提问？(y/n)：")
        if cont.lower() != 'y':
            break


if __name__ == "__main__":
    main()
