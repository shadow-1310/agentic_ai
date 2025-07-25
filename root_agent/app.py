import streamlit as st
import io
import os
import asyncio
import uuid # For generating unique session IDs
import warnings
from tts import synthesize_text
from typing import Optional
from PIL import Image

# Suppress all warnings
warnings.filterwarnings("ignore")

# Import necessary ADK components
# Make sure 'agent.py' containing 'root_agent' is in the same directory
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

# --- ADK Agent Interaction Logic ---

async def get_agent_response_async(runner: Runner, user_id: str, session_id: str, query: str, audio_bytes = None, image = None) -> str:
    """
    Sends a query to the ADK agent and retrieves its final response.
    This function is adapted from your original `call_agent_async`.
    """
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "user", "content": query})

    if audio_bytes:
        # audio_content = types.FileData(
        #                     mime_type='audio/mpeg', # Or 'image/png', etc.
        #                     file_uri=path_audio
        #                 )
        print(type(audio_bytes))
        audio_content = types.Blob(
            mime_type='audio/wav',
            data=audio_bytes,
        )
        # Prepare the user's message in ADK format
        content = types.Content(role='user',parts=[types.Part(inline_data=audio_content)])
    elif image:
        image_content = types.Blob(
            mime_type='image/png', # Or 'image/png', etc.
            data=image
        )
        # Prepare the user's message in ADK format
        content = types.Content(role='user',parts=[types.Part(text=query), types.Part(inline_data=image_content)])
    else:
        content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."

    # Key Concept: run_async executes the agent logic and yields Events.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break # Stop processing events once the final response is found

    return final_response_text

# --- Streamlit App Setup ---

def main():
    st.set_page_config(page_title="ADK Chatbot", layout="centered")
    st.title("ADK Agent Chatbot")
    st.markdown("---")

    # Initialize ADK components in Streamlit's session state
    # This ensures they are created only once per user session
    if "session_service" not in st.session_state:
        st.session_state.session_service = InMemorySessionService()
        st.session_state.app_name = "streamlit_adk_chatbot"
        st.session_state.user_id = "streamlit_user" # A fixed user ID for this simple app
        st.session_state.session_id = str(uuid.uuid4()) # Unique session ID for each run

        # Initialize the session for the ADK runner
        asyncio.run(st.session_state.session_service.create_session(
            app_name=st.session_state.app_name,
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id
        ))

        # IMPORTANT: Ensure root_agent is available.
        # It's assumed 'root_agent' is imported from 'agent.py'
        try:
            from agent import root_agent
            st.session_state.runner = Runner(
                agent=root_agent,
                app_name=st.session_state.app_name,
                session_service=st.session_state.session_service
            )
            st.success("ADK Agent initialized successfully!")
        except ImportError:
            st.error("Error: 'root_agent' not found. Please ensure 'agent.py' is in the same directory and defines 'root_agent'.")
            st.stop() # Stop the app if the agent cannot be loaded
        except Exception as e:
            st.error(f"Error initializing ADK Runner: {e}")
            st.stop()

    # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Audio Input ---
    audio_bytes = st.audio_input(
        label="Record your message",
        key="audio_input_recorder", 
    )
    if audio_bytes:
        audio_bytes = audio_bytes.read()

    uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["png", "jpg", "jpeg", "gif"]
)

    if uploaded_file is not None:
    # To display the file, we first open it using the Python Imaging Library (PIL).
    # The st.image function can then render this image object.
        try:
            image = Image.open(uploaded_file)

            st.success("Image uploaded successfully!")
            st.image(
                image, 
                caption=f"You uploaded: {uploaded_file.name}", 
                use_container_width=True
            )
            st.info(f"Image Details: {image.format} format, {image.size[0]}x{image.size[1]} pixels")
            png_buffer = io.BytesIO()
            image.save(png_buffer, format='PNG')
            png_bytes = png_buffer.getvalue()

        except Exception as e:
            st.error(f"Error: Unable to open or process the image file. Please ensure it's a valid image. Details: {e}")

    # Save the file
    # if audio_bytes:
    #     path_audio = "audio_input.mpeg" 
    #     with open(path_audio, "wb") as f:
    #         f.write(audio_bytes.read())

    # React to user input
    if prompt := st.chat_input("Ask your agent a question...") or audio_bytes:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response asynchronously
        with st.spinner("Agent thinking..."):
            if "runner" in st.session_state:
                response = asyncio.run(
                    get_agent_response_async(
                        st.session_state.runner,
                        st.session_state.user_id,
                        st.session_state.session_id,
                        prompt,
                        audio_bytes,
                        png_bytes,
                    )
                )
            else:
                response = "Agent not initialized. Please check for errors above."

        # Display agent response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.button("🔊 Play Last Response"):
        # Find the last assistant message
        last_assistant_message = None
        for message in reversed(st.session_state.messages):
            if message["role"] == "assistant":
                last_assistant_message = message["content"]
                break

        if last_assistant_message:
            audio_file_path = "last_response.mp3"
            synthesize_text(last_assistant_message, audio_file_path)
            if os.path.exists(audio_file_path):
                with open(audio_file_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")
                # Clean up the temporary file after playing
                os.remove(audio_file_path)
            else:
                st.warning("Could not generate audio for the last response.")
        else:
            st.info("No assistant response to play yet.")

if __name__ == "__main__":
    main()
