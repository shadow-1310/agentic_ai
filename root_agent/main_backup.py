import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from agent import root_agent
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response." # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
    #   print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

    # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():

            # print(event.content)
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
                # final_response_text = event.content.parts[-1].text
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break # Stop processing events once the final response is found
    print(f"<<< Agent Response: {final_response_text}")

async def run_conversation(runner, user_id, session_id):
    await call_agent_async("what is the name of chapter 3 of class 6 NCERT english textbook?",
                                       runner=runner,
                                       user_id=user_id,
                                       session_id=session_id)

    await call_agent_async("Base on KTS textbooks, Kisa Gotami again goes from house to house after she speaks with the Buddha. What does she ask for, the second time around? Does she get it? Why not?",
                                       runner=runner,
                                       user_id=user_id,
                                       session_id=session_id) # Expecting the tool's error message

    await call_agent_async("Generate a few MCQ questions from the chapter Glimses of India from KTS textbook.",
                                       runner=runner,
                                       user_id=user_id,
                                       session_id=session_id)

# Execute the conversation using await in an async context (like Colab/Jupyter)
# await run_conversation()
async def main():
    # --- Session Management ---
    # Key Concept: SessionService stores conversation history & state.
    # InMemorySessionService is simple, non-persistent storage for this tutorial.
    session_service = InMemorySessionService()

    # Define constants for identifying the interaction context
    APP_NAME = "weather_tutorial_app"
    USER_ID = "user_1"
    SESSION_ID = "session_001" # Using a fixed ID for simplicity

    # Create the specific session where the conversation will happen
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    # --- Runner ---
    # Key Concept: Runner orchestrates the agent execution loop.
    runner = Runner(
        agent=root_agent, # The agent we want to run
        app_name=APP_NAME,   # Associates runs with our app
        session_service=session_service # Uses our session manager
    )
    print(f"Runner created for agent '{runner.agent.name}'.")

    await run_conversation(runner, USER_ID, SESSION_ID)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
