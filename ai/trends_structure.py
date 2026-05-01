from typing import List
from pydantic import BaseModel, Field


class ClusterAnalysis(BaseModel):
    cluster_id: int = Field(description="the integer id of the input cluster you are analyzing")
    label: str = Field(description="2-6 word label naming this research direction")
    one_line: str = Field(description="one-sentence description of what this cluster is about")
    existing_landscape: str = Field(description="2-4 sentences naming actual companies, products, or open-source projects that already serve this space; if you don't know any, say so")
    research_industry_gap: str = Field(description="2-4 sentences identifying the specific mismatch between where research is going and what industry has built; be concrete, not generic")
    startup_thesis: str = Field(description="3-5 sentences describing what a startup could build to fill the gap, who would buy it, and why it would be defensible")
    why_now: str = Field(description="1-2 sentences on why this is timely — what trend signal makes this the right moment")
    risks: str = Field(description="1-2 sentences on what could kill this thesis — incumbents, technical risk, market timing")
    confidence: str = Field(description="one of: high, medium, low — how confident are you that this gap is real and the thesis is non-obvious")


class TrendsReport(BaseModel):
    overview: str = Field(description="3-5 sentence summary of what's happening across the corpus this period — convergence patterns, surprising shifts, dominant directions")
    top_clusters: List[ClusterAnalysis] = Field(description="ranked list of the most promising clusters with full gap analysis, ordered best opportunity first")
