from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from src.utils import deduplicate_and_format_sources, tavily_search, format_sources
from src.state import SummaryState, SummaryStateInput, SummaryStateOutput
from src.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions
from langsmith import traceable
import json
import os
from typing_extensions import Literal
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGSMITH_ENDPOINT"] ="https://api.smith.langchain.com"
max_loops = os.getenv("MAX_WEB_RESEARCH_LOOPS")


#Nodes
@traceable
def generate_query(state: SummaryState):
    """ Generate a query for web search """

    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    llm_json_mode = ChatGroq( model_name="deepseek-r1-distill-qwen-32b",
                                   api_key=groq_api_key, temperature=0, 
                                   model_kwargs={"response_format": {"type": "json_object"}})
    
    results = llm_json_mode.invoke([SystemMessage(content=query_writer_instructions_formatted),
                                   HumanMessage(content=f"Generate a query for web search:")])
    
    query = json.loads(results.content)

    question = query.get('query')

    if not question:

        return {"search_query": f"Tell me about {state.research_topic}"}

    return {"search_querry": query['query']}

@traceable
def web_research(state: SummaryState):
    """ Gather information from the web """

    search_results = tavily_search(state.search_querry, include_raw_content=True, max_results=3)
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    sources = format_sources(search_results)

    return {"sources_gathered": [sources], "web_search_results": [search_str], "research_loop_count": state.research_loop_count + 1}

@traceable
def summarize_sources(state:SummaryState):
    """ Summarize the gathered sources """

    existing_summary = state.running_summary
    recent_web_search = state.web_search_results[-1]

    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New search results> \n {recent_web_search} \n <New search results>\n\n"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search results> \n {recent_web_search} \n <Search results>\n\n"
        )

    llm = ChatGroq( model_name="deepseek-r1-distill-llama-70b",
                                   api_key=groq_api_key, temperature=0)
    
    results = llm.invoke([SystemMessage(content=summarizer_instructions),
                          HumanMessage(content=human_message_content)])
    
    running_summary = results.content

    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

@traceable
def reflect_on_summary(state: SummaryState):
    """ Reflect on the summary and generate a follow-up query """

    reflection_instructions_formatted = reflection_instructions.format(research_topic = state.research_topic)
    llm_json_mode = ChatGroq( model_name="deepseek-r1-distill-llama-70b",
                                   api_key=groq_api_key, temperature=0, 
                                   model_kwargs={"response_format": {"type": "json_object"}})
    
    results = llm_json_mode.invoke([SystemMessage(content=reflection_instructions_formatted),
                                    HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")])
    
    follow_up_query = json.loads(results.content)

    query = follow_up_query.get('follow_up_query')

    if not query:

        return {"search_query": f"Tell me more about {state.research_topic}"}

    return {"search_query": follow_up_query['follow_up_query']}

def finalize_summary(state: SummaryState):
    """ Finalize the summary """

    all_sources = "\n".join(source for source in state.sources_gathered)
    running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"

    return {"running_summary": running_summary}

def route_research(state: SummaryState) -> Literal["finalize_summary","web_research"]:
    if state.research_loop_count <= int(max_loops):
        return "web_research"
    else:
         return "finalize_summary"
    
#Add nodes
workflow = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput)
workflow.add_node("generate_query",generate_query)
workflow.add_node("web_research",web_research)
workflow.add_node("summarize_sources",summarize_sources)
workflow.add_node("reflect_on_summary",reflect_on_summary)
workflow.add_node("finalize_summary",finalize_summary)

#Add edges
workflow.add_edge(START,"generate_query")
workflow.add_edge("generate_query","web_research")
workflow.add_edge("web_research","summarize_sources")
workflow.add_edge("summarize_sources","reflect_on_summary")
workflow.add_conditional_edges("reflect_on_summary",route_research)
workflow.add_edge("finalize_summary",END)

app = workflow.compile()

if __name__=="__main__":

    input_data = SummaryStateInput(research_topic="Latest advancements in LLM agents and there usecases")
    output = app.invoke(input_data)
    print(output["running_summary"])