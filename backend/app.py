# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain.memory import ConversationBufferMemory
from dataprocess import return_answer, load_default_resources, default_resources, generate_email_format_answer, check_response_before_answer, translate_to_selected_response_language, load_memory
import os
from audio_recorder_streamlit import audio_recorder
from tempfile import NamedTemporaryFile
import whisper
import time
from pydub import playback
# import pydub
import urllib3
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings("ignore", category=DeprecationWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from openai import OpenAI
# read the api key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sender_email=os.getenv("sender_email")
password=os.getenv("password")


st.set_page_config(page_title = "San Francisco Bay University",
                   page_icon= "./images/chatbot.png",
                   initial_sidebar_state="auto",
                    )


openai_models = ["gpt-3.5-turbo-1106","ft:gpt-3.5-turbo-1106:learninggpt:sfbu-bot:9CVU8Zib"]
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

default_max_tokens = 500

###### Header UI Start ###################################
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image('./images/SFBU-logo_0.png')
    st.header("SFBU ChatBot", anchor=False, divider="blue")

##########################################################


######  Default State Variable #####################

@st.cache_resource
def load_resources():
    return load_default_resources(load_from_local_stored_files=True)
# Load default resources
if 'default_vectorstore' not in st.session_state:
    st.session_state.default_vectorstore = load_resources()

# Initialise default retriver variable
if 'retriever' not in st.session_state:
    st.session_state.retriever = st.session_state.default_vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5})


# Initialise question_submit_clicked variable
if 'question_submit_clicked' not in st.session_state:
    st.session_state.question_submit_clicked = False

# Initialise answer_in_email_format_clicked variable
if 'answer_in_email_format_clicked' not in st.session_state:
    st.session_state.answer_in_email_format_clicked = False

# # Initialise chat history variable
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# Initialise model variable
if 'model' not in st.session_state:
    st.session_state.model = openai_models[0]

# Initialise temperature variable
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.0

if 'audio_recorder_key' not in st.session_state:
    st.session_state.audio_recorder_key = "1"

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = load_whisper_model()


# Initialise default response language variable
# The first one is Default, which means the default response language is same as the input language
response_languages = ["Default", "English","Hindi",
                      "Spanish", "French", "Chinese", 
                      "Arabic","Japnese","Vietnamese",
                      "Nepali","Ukranian","Icelandic", 
                      "Indonesian", "Italian", "Kannada",
                        "Kazakh", "Korean", "Latvian", 
                        "Lithuanian", "Macedonian","Punjabi",
                        "Afrikaans", "Arabic", "Armenian", 
                        "Azerbaijani", "Belarusian", "Bosnian", 
                        "Bulgarian", "Catalan","German",
                        "Marathi","Bengali","Malyalam"]

response_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
if 'response_language' not in st.session_state:
    st.session_state.response_language = response_languages[0]

if 'response_voice' not in st.session_state:
    st.session_state.response_voice = response_voices[0]

##########################################################


##### Sidebar UI Start ###################################

with st.sidebar:
    cl1, cl2, cl3 = st.columns([1, 2, 1])
    with cl2:
        st.image('./images/chatbot.png')

    st.markdown("# SFBU BOT CONFIGURATION")
    st.divider()
    st.write('Choose OpenAI Model')
    # default to the first model
    # Add a label to avoid the warning and use label_visibility="collapsed" to hide the label
    model = st.selectbox('Select your model',
                         openai_models, index=openai_models.index(st.session_state.model), label_visibility="collapsed")
    with st.container():
        st.markdown(
            """<div>
                <div><small>GPT-3.5 : Base Model (Less customised)</small></div>
                <div><small>gpt-3.5-turbo-1106:learninggpt:sfbu-bot:9CVU8Zib : Fine tuned model</small></div>
                </div>
            """, unsafe_allow_html=True)

    # set selected value back to session state
    st.session_state.model = model
    print(f"st.session_state.model: {st.session_state.model}")

    st.divider()
    # Check session state first
    st.write('Choose Temperature')
    # default to the default temprature
    # Add a label to avoid the warning and use label_visibility="collapsed" to hide the label
    temperature = st.slider('Select your Temperature', min_value=0.0,
                            max_value=1.0, step=0.01, value=st.session_state.temperature, label_visibility="collapsed")

    # with st.container():
    #     st.markdown(
    #         """<div>
    #             <div><small>Lower temperature : more deterministic results, higher accuracy</small></div>
    #             <div><small>Higher temperature : more creative results, lower accuracy</small></div>
    #             </div>
    #         """, unsafe_allow_html=True)

    # # set selected value back to session state
    st.session_state.temperature = temperature
    # # print(f"st.session_state.temperature: {st.session_state.temperature}")

    st.divider()
    st.write('Choose Response Language')
    response_language = st.selectbox('Select your response language',
                                     response_languages, index=0, label_visibility="collapsed")
    with st.container():
        st.markdown(
            """<div>
                <div><small>By default, the response language is same as the input language.</small></div>
                </div>
            """, unsafe_allow_html=True)

    # set selected value back to session state
    st.session_state.response_language = response_language
    # print(f"st.session_state.response_language: {st.session_state.response_language}")

    st.divider()
    st.write('Choose Voice')
    response_voice = st.selectbox('Select your voice',
                                     response_voices, index=0, label_visibility="collapsed")
    
    with st.container():
        st.markdown(
            """
            """, unsafe_allow_html=True)

    # set selected value back to session state
    st.session_state.response_voice = response_voice
##########################################################


def generate_answer():
    st.session_state.question_submit_clicked = True


def generate_answer_in_email():
    st.session_state.answer_in_email_format_clicked = True


def result_all_button_state():
    st.session_state.question_submit_clicked = False
    st.session_state.answer_in_email_format_clicked = False
    st.session_state.query = ''


def clear_chat_history():
    st.session_state.messages = []

# def send_email():
#     st.session_state.messages = []
   
def send_email(sender_email, sender_password, recipient_email, subject, body):
    # Create the MIME object
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject

    # Add body text to the email
    message.attach(MIMEText(body, "plain"))

    try:
        # Connect to the SMTP server (in this case, Gmail's SMTP server)
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            # Start TLS for security'\
            server.ehlo()
            server.starttls()
            # Login to the email account
            server.login(sender_email, sender_password)
            # Send the email
            server.sendmail(sender_email, recipient_email, message.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")
if st.session_state.response_voice != "Default":
            voice_=st.session_state.response_voice

def generate_audio(input):

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice_,
        input=input
    )

    response.stream_to_file("output.mp3")


    st.audio("output.mp3")

##### Main UI and Logic ###################################



if 'text_received' not in st.session_state:
    st.session_state.text_received = ""



##########################

##### Textbox & Question Submit Button #######
label1=st.markdown("""<div style="text-align: center;">
                                        <div>Hey there! Hope you are doing well today. I would like to welcome you to SFBU.</div> 
                                        <div>I'm your personal assistant. How can I help you?</div>
                                        </div>""",unsafe_allow_html=True)
st.session_state.query = st.text_input(label=" ",
                                       type="default", autocomplete="off", value=st.session_state.text_received)

audio_bytes = audio_recorder(text="Click to start or stop recording", pause_threshold=1.5, key="audio",
                             sample_rate=60000, energy_threshold=0.003, icon_size="2x", recording_color="#000000",neutral_color="#666666")
text = ""
if audio_bytes is not None:
    # Save the audio bytes to a temporary file
    with NamedTemporaryFile(delete=False, suffix='.wav') as f:
        f.write(audio_bytes)
        temp_audio_path = f.name
    # Load the Whisper model and transcribe the audio file
    model_audio = st.session_state.whisper_model
    result_text = model_audio.transcribe(temp_audio_path)
    text = result_text["text"]
if text:
    st.session_state.text_received = text

c1, c2, c3,c4 = st.columns([3, 3, 3,3])
with c1:
    st.button(label="💬", type="primary",
              disabled=False, use_container_width=True, on_click=generate_answer,help="Click to generate answer")
with c2:
    st.button(label="📧", type="primary",
              disabled=False, use_container_width=True, on_click=generate_answer_in_email,help="Click to generate answer in email format")
with c3:
    st.button(label="🗑️", type="primary",
              disabled=False, use_container_width=True, on_click=clear_chat_history,help="Click to clear chat history")

with c4:
    if st.button(label="📩",type="primary",disabled=False, use_container_width=True,help="Click to send an email"):
        temp_messages = st.session_state.messages.copy()
        final_response = generate_email_format_answer(
                    client, temp_messages, model, temperature)
        subject = ""
        content = ""

        # Split the string based on "Subject:" and "Content:"
        sections = final_response.split("\n")

        # Iterate over each section and extract subject and content
        for line in sections:
            # Strip whitespace from the line
            line = line.strip()
            
            # Check if the line starts with "Subject:"
            if line.startswith("Subject:"):
                subject = line[len("Subject:"):].strip()
            # Check if the line starts with "Content:"
            elif line.startswith("Content:"):
                # Append content from the line to the content string
                content += line[len("Content:"):].strip() + "\n"

        # Remove the last newline character from the content string
        content = content.rstrip()

        # Print the extracted subject and content
        print("Subject:", subject)
        print("Content:", content)
        send_email(sender_email,password,"prachi1615@gmail.com",subject,content)


st.session_state.qa = return_answer(
        st.session_state.temperature, st.session_state.model, st.session_state.retriever)
    
memory = load_memory(st)

# After submitting the question
if st.session_state.question_submit_clicked or st.session_state.answer_in_email_format_clicked:


    query = st.session_state.query

    print("User submitted question --> : ", query)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Generating Answer..."):
        # Call the QA function with the necessary parameters to retrieve the initial resposne
        first_result = st.session_state.qa(
            {
                "question": query,
                "chat_history": memory.load_memory_variables({})["history"]
            }
        )

        first_result = first_result["answer"]

        print("First result --> : ", first_result)

        if st.session_state.question_submit_clicked:

            final_response = check_response_before_answer(
                client, query, first_result, model, temperature, default_max_tokens)

            print("check_response_before_answer, final result --> : ", final_response)

        elif st.session_state.answer_in_email_format_clicked:
            temp_messages = st.session_state.messages.copy()
            temp_messages.append({"role": "assistant", "content": first_result})
            final_response = generate_email_format_answer(
                client, temp_messages, model, temperature)

            print("generate_email_format_answer, final result --> : ", final_response)

        if st.session_state.response_language != "Default":
            final_response = translate_to_selected_response_language(client,
                                                                     final_response, st.session_state.response_language, model, temperature)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = final_response
        full_response = '<div>' + full_response + '</div>'
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(
                full_response + "|", unsafe_allow_html=True)
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.divider()
        if st.session_state.question_submit_clicked:
            with st.spinner("Generating Audio Message..."):
                generate_audio(final_response)
        else:
            pass

    st.session_state.messages.append(
        {"role": "assistant", "content": final_response})


    result_all_button_state()


