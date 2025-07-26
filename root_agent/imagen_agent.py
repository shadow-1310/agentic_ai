from tools.imagen_prompt import IMAGEGEN_PROMPT
from google.adk.agents import Agent
from tools.image_generation_tool import generate_images

imagen_agent_tool = Agent(
    name="imagen_agent_tool",
    model="gemini-2.5-flash",
    description=("You are an expert in creating images with imagen 3"),
    instruction=(IMAGEGEN_PROMPT),
    tools=[generate_images],
    # output_key="output_image",
)

