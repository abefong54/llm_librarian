import asyncio
import helper_models.sk_helper as sk
import helper_models.langchain_helper as lh


audio_file1 = "./data/audio_file_1.mp3"
audio_file2 = "./data/audio_file_2.mp3"
book_csv = "./data/book_dataset.csv"
small_book_csv = "./data/book_dataset_small.csv"

async def main():
    # RUN THE SK MODEL UI
    # ui.run_ui()

    # RUN THE SK MODEL WHISPER FEATURE
    # audio = "./data/audio_file_1.mp3"


    # try using this website instead
    # https://www.govinfo.gov/app/collection/bills/118/hconres/all/%7B%22pageSize%22%3A%2220%22%2C%22offset%22%3A%220%22%7D

    
    user_query = ""
    pdf_url = "https://www.congress.gov/118/bills/hr4366/BILLS-118hr4366eah.pdf"
    qa = lh.summarize_article_from_pdf_vector(pdf_url)

    while(user_query != "exit"):
        user_query = input("\nAsk a question about the document:\n\t")
        output = qa.run(user_query)
        print("Query:" + user_query)
        print("\nAnswer: \n " + output)
        print("########################")



    # # # USING SEMANTIC KERNEL
    # if user_request:
    #     rec = sk.recommend_books_based_on_request(user_request=user_request)


    # USING LANGCHAIN
    # if user_request:
    #     print(user_request)
    #     rec = lh.recommend_book_from_csv_vector(user_query=user_request)
    # print(rec)

# NOTE THAT KERNEL RUNS ASYNCHRONOUSLY
if __name__ == "__main__":
    asyncio.run(main())