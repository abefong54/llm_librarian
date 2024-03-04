import os
import openai
import semantic_kernel as sk
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

load_dotenv()


# SET UP THE KERNEL
openai.api_key = os.getenv("OPENAI_API_KEY")
kernel = sk.Kernel()
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-4",api_key, org_id))
prompt = """
You are a librarian. Provide a recommendation to a book based on the following information: {{$input}}.
Explain your thinking step by step, including a list of top books you selected and how you got to your answer.
Please format your output by puttnig new line character after each step.
"""


def process_user_audio(path_to_file=""):
    with open(path_to_file, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript.text

def recommend_books_based_on_request(user_request):
    summarize = kernel.create_semantic_function(prompt,max_tokens=512)
    result = summarize(user_request)
    return str(result)



def librarian_app(name="", genre="", description=""):
    response = ""
    if name is not "":
        response = recommend_books_based_on_request(name)
    elif genre is not "":
        response = recommend_books_based_on_request(genre)
    elif description is not "":
        response = recommend_books_based_on_request(description)

    return response

