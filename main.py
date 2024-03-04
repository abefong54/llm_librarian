import os
import openai
import asyncio
import semantic_kernel as sk
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
kernel = sk.Kernel()
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-4",api_key, org_id))
audio_file = "./audio_file_2.mp3"

def process_user_audio(path_to_file=""):
    with open(path_to_file, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        # print(transcript)
        return transcript.text

def recommend_books_based_on_request(user_request):
    # CREATE FUNCTION
    summarize = kernel.create_semantic_function("Recommend me 3 books based on this request: {{$input}}\n\n Tell me what you were asked for first and then a TLDR with why you chose this recommendation in less than 10 words.")
    print(summarize(user_request))


def librarian_app(audio_file):
    user_request = process_user_audio(audio_file)
    recommend_books_based_on_request(user_request)

# NOTE THAT KERNEL RUNS ASYNCHRONOUSLY
async def main():
    librarian_app(audio_file)

if __name__ == "__main__":
    asyncio.run(main())


