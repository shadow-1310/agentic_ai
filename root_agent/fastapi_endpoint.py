import io
import base64
from typing import Tuple, Optional
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import uuid
import warnings
from typing import Optional
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Suppress all warnings
warnings.filterwarnings("ignore")

# Import necessary ADK components
# Make sure 'agent.py' containing 'root_agent' is in the same directory
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

# Assuming 'tts.py' contains the synthesize_text function
from tts import synthesize_text
from agent import root_agent

app = FastAPI(
    title="ADK Agent FastAPI",
    description="A FastAPI application for interacting with an ADK Agent, supporting text, audio, and image inputs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# In-memory storage for session management (for demonstration purposes)
# In a production environment, consider a persistent store like Redis or a database
class SessionManager:
    def __init__(self):
        self.sessions = {} # user_id -> {session_id: runner}
        self.session_service = InMemorySessionService()

    async def get_or_create_runner(self, user_id: str, session_id: str) -> Runner:
        if user_id not in self.sessions:
            self.sessions[user_id] = {}

        if session_id not in self.sessions[user_id]:
            app_name = "fastapi_adk_chatbot"
            
            await self.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id
            )
            
            runner = Runner(
                agent=root_agent,
                app_name=app_name,
                session_service=self.session_service
            )
            self.sessions[user_id][session_id] = runner
            print(f"Created new session and runner for user: {user_id}, session: {session_id}")
        return self.sessions[user_id][session_id]

session_manager = SessionManager()

async def get_agent_response_async(runner: Runner, user_id: str, session_id: str, query: str, audio_bytes: Optional[bytes] = None, image_bytes: Optional[bytes] = None):
    """
    Sends a query to the ADK agent and retrieves its final response.
    """
    if audio_bytes:
        audio_content = types.Blob(
            mime_type='audio/wav',
            data=audio_bytes,
        )
        content = types.Content(role='user', parts=[types.Part(inline_data=audio_content)])
    elif image_bytes:
        print(type(image_bytes))
        image_content = types.Blob(
            mime_type='image/png',
            data=image_bytes
        )
        content = types.Content(role='user', parts=[types.Part(text=query), types.Part(inline_data=image_content)])
    else:
        content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        print(f"ADK Event: {event}")
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    # Check for generated image and clean up
    target_image = "generated_image_0.png"
    if os.path.exists(target_image):
        try:
            with open(target_image, "rb") as image_file:
                image_bytes = image_file.read()
            encoded_bytes = base64.b64encode(image_bytes).decode()
            print(f"Found generated image: {target_image}. Deleting...")
            os.remove(target_image)
        except Exception as e:
            print(f"Error removing generated image {target_image}: {e}")

    print("type of generated image_bytes: ", type(image_bytes))
    print("type of encoded image_bytes: ", type(encoded_bytes))
    print("type of text is: ", type(final_response_text))
    # return final_response_text, encoded_bytes
    return {
        "text": final_response_text,
        "bytes_base64": encoded_bytes
    }

# --- FastAPI Endpoints ---

class ChatRequest(BaseModel):
    query: Optional[str] = None
    user_id: str = "default_user"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    query: Optional[str] = Form(None),
    user_id: str = Form("default_user"),
    session_id: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    """
    Endpoint for chatting with the ADK Agent using text, audio, or image input.
    """
    if not query and not audio_file and not image_file:
        raise HTTPException(status_code=400, detail="Either 'query', 'audio_file', or 'image_file' must be provided.")

    if session_id is None:
        session_id = str(uuid.uuid4())
    
    runner = await session_manager.get_or_create_runner(user_id, session_id)

    audio_bytes = None
    if audio_file:
        audio_bytes = await audio_file.read()

    image_bytes = None
    if image_file:
        # FastAPI handles file uploads in memory or as temporary files.
        # We need to read the bytes and potentially convert to PNG if not already.
        try:
            image_data = await image_file.read()
            # Attempt to open as PIL Image to ensure it's a valid image and convert to PNG
            img = Image.open(io.BytesIO(image_data))
            png_buffer = io.BytesIO()
            img.save(png_buffer, format='PNG')
            image_bytes = png_buffer.getvalue()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    response_text, image_bytes = await get_agent_response_async(
        runner,
        user_id,
        session_id,
        query or "", # Pass empty string if query is None for audio/image inputs
        audio_bytes,
        image_bytes
    )
    return ChatResponse(response=response_text, session_id=session_id)

@app.post("/synthesize_speech")
async def synthesize_speech(text: str = Form(...)):
    """
    Synthesizes speech from the given text and returns an audio file.
    """
    audio_file_path = "synthesized_speech.mp3"
    try:
        synthesize_text(text, audio_file_path)
        if os.path.exists(audio_file_path):
            def iterfile():
                with open(audio_file_path, "rb") as f:
                    yield from f
                os.remove(audio_file_path) # Clean up the file after streaming

            return StreamingResponse(iterfile(), media_type="audio/mpeg")
        else:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {e}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # You might want to adjust the host and port for deployment
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, port=8000)
