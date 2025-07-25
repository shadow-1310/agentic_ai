from google.adk.agents import Agent
from google.adk.tools import google_search, VertexAiSearchTool 
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

from dotenv import load_dotenv
# from .prompts import return_instructions_root
import os

load_dotenv()

ncert_retrieval = VertexAiRagRetrieval(
    name='retrieve ncert textbook',
    description=(
        'Use this tool to retrieve documentation and reference materials for the question from the NCERT Textbook corpus,'
    ),
    rag_resources=[
        #NCERT Textbooks
        rag.RagResource(
            rag_corpus="projects/265110558107/locations/us-central1/ragCorpora/576460752303423488"
        ),
    ],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
)

kts_retrieval = VertexAiRagRetrieval(
    name='retrieve kts textbook',
    description=(
        'Use this tool to retrieve documentation and reference materials for the question from the KTS Textbook corpus,'
    ),
    rag_resources=[
        # KTS Textbooks
        rag.RagResource(
            rag_corpus="projects/265110558107/locations/us-central1/ragCorpora/5764607523034234880"
        ),
    ],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
)

# vertexai_search_tool = VertexAiSearchTool(
#    data_store_id="projects/tough-nature-466516-r4/locations/global/collections/default_collection/dataStores/YOUR_DATA_STORE_ID"
# )

rag_agent_ncert = Agent(
    name="rag_agent_ncert",
    model="gemini-2.5-flash",  
    # model="gemini-2.0-flash",  
    description="Agent to answer questions using RAG on diffferent Textbooks.",
    instruction="You are an expert researcher. You always stick to the facts.",
    tools=[ncert_retrieval]
)

rag_agent_kts = Agent(
    name="rag_agent_kts",
    model="gemini-2.5-flash",  
    # model="gemini-2.5-flash",  
    description="Agent to answer questions using RAG on diffferent Textbooks.",
    instruction="You are an expert researcher. You always stick to the facts.",
    tools=[kts_retrieval]
)
