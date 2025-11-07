from langchain_openai import ChatOpenAI
import os 
#client = httpx.Client(verify=False)
OpenAiKey = "sk-JEVyjAuuQSr7akfYgASCXA"
llm = ChatOpenAI(
base_url="https://genailab.tcs.in",
model = "azure_ai/genailab-maas-DeepSeek-V3-0324",
api_key= sk-JEVyjAuuQSr7akfYgASCXA, # Will be provided during event. And this key is for 
#Hackathon purposes only and should not be used for any unauthorized 
#purposes
#
#http_client = client
)
llm_Response = llm.invoke("Hi")
print(llm_Response)
