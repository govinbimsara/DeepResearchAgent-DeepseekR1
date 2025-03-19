import operator
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated

@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)
    search_querry: str = field(default=None)
    web_search_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)
    running_summary: str = field(default=None)

@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)

