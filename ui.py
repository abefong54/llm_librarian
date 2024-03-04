import streamlit as st
import sk_helper as sk
import textwrap


def run_ui():
    st.title("Librarian-GPT")
    with st.sidebar: 
        with st.form(key="my_form_2"):

            name_of_last_book = st.sidebar.text_area(
                label = "TITLE of book did you last read?",
                max_chars = 50,
            )
            submit_button_2 = st.form_submit_button(label="Submit")
       
        # with st.form(key="my_form_1"):
        #     genre_of_last_book = st.sidebar.text_area(
        #         label = "GENRE of the last book you read?",
        #         max_chars = 50
        #     )
        #     submit_button_1 = st.form_submit_button(label="Submit")
     

        # with st.form(key="my_form_3"):
        #     description_of_book = st.sidebar.text_area(
        #         label = "SHORT DESCRIPTION of books you liked in the past?",
        #         max_chars = 50
        #     )
        #     submit_button_3 = st.form_submit_button(label="Submit")

    # if name_of_last_book or genre_of_last_book or description_of_book:
        # db = lh.create_vector_db_from_youtube_url(youtube_url)
    response=""
    if name_of_last_book:
        response = sk.librarian_app(name=name_of_last_book)
        # elif genre_of_last_book:
        #     response = sk.librarian_app(genre=genre_of_last_book)
        # elif description_of_book:
        #     response = sk.librarian_app(description=description_of_book)
    st.text(textwrap.fill(response, width = 100))

