from operator import add
from typing import Annotated, List, NotRequired, Optional, TypedDict


class PersonalAIState(TypedDict):
    task: str
    post_to_telegram: bool 
    search_queries: List[str]
    search_results: List[dict]
    scraped_content: List[str]
    research_summary: str
    thread: List[str]
    twitter_thread: NotRequired[List[str]]
    final_status: str
    messages: Annotated[List[dict], add]
    error: Optional[str]
    current_agent: str
