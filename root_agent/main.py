# main.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Assuming 'agent.py' exists in the same directory and defines 'root_agent'
# If 'root_agent' is not directly importable or needs specific initialization,
# you might need to adjust this import or its instantiation.
try:
    from agent import root_agent
except ImportError:
    print("Warning: 'agent.py' not found or 'root_agent' not defined within it. Please ensure 'agent.py' is in the same directory and contains 'root_agent'.")
    # As a fallback for demonstration, create a dummy agent if root_agent isn't found
    # In a real scenario, you'd want to ensure your agent is properly loaded.
    class DummyAgent(Agent):
        def __init__(self):
            super().__init__(name="DummyAgent")
            self.model = LiteLlm(model_name="gemini-pro") # Placeholder model

        async def call_agent(self, context):
            # Simple echo for dummy agent
            user_message = context.get_message().text
            if user_message:
                await context.send_message(f"Dummy Agent received: {user_message}")
            else:
                await context.send_message("Dummy Agent received an empty message.")
    root_agent = DummyAgent()


# --- Global variables for agent runner and session service ---
session_service: InMemorySessionService = None
runner: Runner = None
APP_NAME = "adk_fastapi_agent" # A unique name for your application

app = FastAPI(
    title="ADK Agent FastAPI",
    description="Exposes Google ADK Agent functionality via a REST API.",
    version="1.0.0",
)

# Configure CORS middleware to allow requests from any origin
# In a production environment, you should restrict origins to your frontend URL(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Request Body Model ---
class ChatRequest(BaseModel):
    query: str
    user_id: str = "default_user" # Provide a default or make it optional
    session_id: str = "default_session" # Provide a default or make it optional

# --- Agent Initialization on Startup ---
@app.on_event("startup")
async def startup_event():
    """Initializes the ADK session service and runner when the FastAPI app starts."""
    global session_service, runner
    print("Initializing ADK Agent components...")
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    print(f"ADK Agent Runner initialized for agent '{runner.agent.name}'.")

# --- Modified call_agent_async to return response ---
async def call_agent_async_for_api(query: str, runner: Runner, user_id: str, session_id: str) -> str:
    """
    Sends a query to the agent and returns the final response text.
    This version is adapted to return the response instead of printing it.
    """
    print(f"\n>>> User Query: {query} (User: {user_id}, Session: {session_id})")

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    # Ensure a session exists for the given user_id and session_id
    # The runner's run_async will create one if it doesn't exist, but explicitly
    # creating it here ensures it's ready for the first interaction.
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id
    )

    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
    except Exception as e:
        print(f"Error during agent run: {e}")
        final_response_text = f"An internal error occurred while processing your request: {e}"

    print(f"<<< Agent Response: {final_response_text}")
    return final_response_text

# --- FastAPI Endpoint ---
@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    API endpoint to send a message to the ADK agent and get a response.
    """
    if not runner:
        raise HTTPException(status_code=503, detail="Agent runner not initialized. Please try again later.")

    try:
        response_text = await call_agent_async_for_api(
            query=request.query,
            runner=runner,
            user_id=request.user_id,
            session_id=request.session_id
        )
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent response: {e}")

# --- Root Endpoint (Optional, for health check) ---
@app.get("/")
async def read_root():
    return {"message": "ADK Agent FastAPI is running!"}

# Instructions to run the application:
# 1. Save the code above as `main.py`.
# 2. Make sure you have your `agent.py` file in the same directory.
# 3. Install necessary libraries: `pip install fastapi uvicorn pydantic google-generativeai google-adk`
# 4. Run the application from your terminal: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
#    - `--reload` is useful for development, it restarts the server on code changes.
# 5. Access the API:
#    - Health check: Go to `http://127.0.0.1:8000/` in your browser.
#    - Interactive API docs: Go to `http://127.0.0.1:8000/docs` (Swagger UI) or `http://127.0.0.1:8000/redoc` (ReDoc).
#    - To test the chat endpoint, use the Swagger UI at `/docs` or a tool like `curl` or Postman:
#      curl -X POST "http://127.0.0.1:8000/chat" \
#      -H "Content-Type: application/json" \
#      -d '{"query": "what is the name of chapter 3 of class 6 NCERT english textbook?", "user_id": "test_user", "session_id": "test_session_001"}'