
import os
from dotenv import load_dotenv
import certifi

from typing import TypedDict, Annotated
import operator
import uuid
import psycopg
from psycopg.rows import dict_row

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# DATABASE CONFIGURATION

def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# GROQ CONFIGURATION

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to .env")


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# TRAVEL STATE

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    final_answer: str
    llm_calls: int


# FLIGHT AGENT

def flight_agent(state: TravelState):

    query = state["user_query"]

    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched successfully.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# HOTEL AGENT

def hotel_agent(state: TravelState):

    query = f"Best hotels for {state['user_query']}"

    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched successfully.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def itineary_agent(state: TravelState):

    prompt = f"""
You are an expert international travel planner.

Create a complete and practical travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{str(state['flight_results'])[:8000]}

Hotel Results:
{str(state['hotel_results'])[:8000]}

Requirements:

1. Create a day-by-day travel itinerary.
2. Include important tourist attractions.
3. Suggest practical transportation options.
4. Include estimated sightseeing costs.
5. Consider the user's total budget.
6. Suggest an appropriate hotel location.
7. Avoid unrealistic travel schedules.
8. Include useful travel tips.

Make the itinerary practical, budget-aware and easy to follow.

Do not invent confirmed flight or hotel bookings.
"""

    response = llm.invoke(prompt)

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# FINAL AGENT
def final_agent(state: TravelState):

    final_prompt = f"""
You are a professional AI travel planning assistant.

Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{str(state['flight_results'])[:8000]}

Hotels:
{str(state['hotel_results'])[:8000]}

Itinerary:
{state['itinerary']}

Format the final answer using these sections:

1. Travel Summary
2. Flight Information
3. Hotel Suggestions
4. Day by Day Itinerary
5. Estimated Budget
6. Final Recommendation

Important instructions:

- Be clear and practical.
- Keep the response within the user's budget whenever possible.
- Mention when live flight prices are unavailable.
- Do not claim that a flight or hotel is booked.
- Distinguish estimated costs from confirmed prices.
- Include international travel considerations.
- Use readable headings and bullet points.
- Avoid repeating information unnecessarily.
"""

    response = llm.invoke(final_prompt)

    return {
        "messages": [response],
        "final_answer": response.content,
        "llm_calls": state.get("llm_calls", 0) + 1
    }
# BUILD LANGGRAPH

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itineary_agent)
graph.add_node("final_agent", final_agent)


graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# POSTGRESQL CHECKPOINTER

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)

checkpointer.setup()

travel_graph = graph.compile(
    checkpointer=checkpointer
)


# RUN TRAVEL AGENT

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "final_answer": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result.get("final_answer", "")

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0)
    }
