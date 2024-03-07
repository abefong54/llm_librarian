from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv
load_dotenv()

########################################################################
########################################################################
# BASIC USE
########################################################################
########################################################################

# UNCOMMENT TO TEST

# # CREATE OUR MODEL
# chat_model = ChatOpenAI(temperature=0)

# # configure output parser
# output_parser = CommaSeparatedListOutputParser()
# format_instructions = output_parser.get_format_instructions()

# # CREATE A PROMPT TEMPLATES
# system_template = """
# You are a helpful librarian that generates comma separated lists on books about {subject}. 
# A user will pass in a category and you should generated 5 book titles in that category in a
# comma separated list . Only return a comma separated list and nothing more \n {format_instructions}
# """
# prompt = PromptTemplate(
#     template=system_template,
#     input_variables=['subject'],
#     partial_variables={"format_instructions":format_instructions}
# )

# # CREATE A CHAIN
# # we can use LangChains parser object, CommaSeparatedListOutputParser
# # https://python.langchain.com/docs/modules/model_io/output_parsers/

# # create the chain
# chain = prompt | chat_model | output_parser
# print(chain.invoke({"subject": "cows"}))

########################################################################
########################################################################
########################################################################

########################################################################
########################################################################
# USING DOCUMENT LOADERS
########################################################################
########################################################################


# UNCOMMENT TO TEST
# from langchain.llms import OpenAI
# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains import RetrievalQA

# url = "https://en.wikipedia.org/wiki/Tea"
# loader = WebBaseLoader(url)
# documents = loader.load() # stores the data from the website
# text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
# texts = text_splitter.split_documents(documents) # splits the data into chunks


# embeddings = OpenAIEmbeddings() # where we can store the embeddngs of the data
# docsearch  = FAISS.from_documents(texts,embeddings) # FAIIS is a product that does embeddings/similarity (semantic) search for us

# qa = RetrievalQA.from_chain_type(
#     llm=OpenAI(),
#     chain_type="stuff",
#     retriever=docsearch.as_retriever())

# while True:
#     query = input("\nAsk a question about tea\n")
#     print(qa.run(query))


########################################################################
########################################################################
# EXTRACTING KEY INFORMATION FROM OUR QUERY
#  WE CAN STORE RESULTS INTO OBJECTS!
########################################################################
########################################################################


# from langchain.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field, validator
# from langchain.output_parsers import PydanticOutputParser

# class Furniture(BaseModel):
#     type: str = Field(description="Type of furniture")
#     style: str = Field(description="Style of furniture")
#     color: str = Field(description="Color of furniture")


# furniture_request = "I would like a red mid century chair."


# # SET UP PARSER
# parser = PydanticOutputParser(pydantic_object=Furniture)


# # SET UP PROMPT TEMPLATE
# prompt = PromptTemplate(
#     template="Answer the user query: \n{format_instructions}\n{query}",
#     input_variables=["query"],
#     partial_variables={"format_instructions":parser.get_format_instructions()})

# _input = prompt.format_prompt(query=furniture_request)
# model = ChatOpenAI(temperature=0)
# output = model.predict(_input.to_string())
# parsed = parser.parse(output)


# # NOTE THAT IT WAS PARSED BUT WITH SOME CONFUSION!
# print(parsed.type)
# print(parsed.style)
# print(parsed.color)
########################################################################
########################################################################
########################################################################

########################################################################
########################################################################
# LIBRARIAN VECTOR DB CHALLENGE
	# - extend librarian app to include a vector db
	# 		- work with CSV data
	# 		- fetch results with relevant book data/meta data
	# 		- provides recommendation and explains why
	# 		- web/console app
########################################################################
########################################################################

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# create a model
# parse in csv data and store into a vector db
# create semantic search on it

def split_and_create_embedding_vector(document_data):
    # VECTOR SET UP
    # split the text for embedding
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document_data) # splits the data into chunks

    # create data vector db
    embeddings = OpenAIEmbeddings() # where we can store the embeddngs of the data
    docsearch = FAISS.from_documents(texts,embeddings) # FAIIS is a product that does embeddings/similarity (semantic) search for us
    return docsearch
   

def summarize_article_from_pdf_vector(pdf_url):
    # configure document loader, CSV
    # url = pdf_url
    # loader = WebBaseLoader(url)

    loader = PyPDFLoader(pdf_url)
    pages = loader.load_and_split()
    pdf_document_vector = split_and_create_embedding_vector(pages)
    # pip install -U langchain-openai


   # prepare the chain,RetrievalQA : Chain for question-answering against an index.
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=pdf_document_vector.as_retriever())

    return qa_chain


def recommend_book_from_csv_vector(user_query):

    # VECTOR SET UP
    # configure document loader, CSV
    library = "./data/book_dataset.csv"
    loader = CSVLoader(file_path=library)
    data = loader.load()

    document_vector = split_and_create_embedding_vector(data)
   
   
   # prepare the chain,RetrievalQA : Chain for question-answering against an index.
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=document_vector.as_retriever())
    
    # SET PROMPT
    book_request_prompt = """
        You are a librarian. Provide a recommendation to a book based on the following request from the user: {user_query}.
        Explain your thinking step by step, including a list of top books you selected and how you got to your answer.
        Please format your output by puttnig new line character after each step.
    """

    output = qa_chain({"query":book_request_prompt})
    return output['result']



# def recommend_book_from_csv_vector_advanced(user_query):

#     output = ""


#     return output


# output_parser = CommaSeparatedListOutputParser()
# format_instructions = output_parser.get_format_instructions()


# prepare a prompt

# create the chain

# return the result
