#!/usr/bin/env python
# coding: utf-8

"""
Streamlit Web Application for Resume Search Platform

This module provides a Streamlit-based web interface for searching and analyzing
candidate resumes. It includes RAG-based search, candidate suggestion, analytics,
and resume upload/parsing functionality.

Author: Siddharth Singh
"""

import logging
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import hashlib
import re

# ChromaDB & RAG
import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor

# Import parser components (LangChain-based)
from resume_parser_langchain import ResumeParser, ChromaDBStore

# Import configuration
from config import get_config, AppConfig

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# RAG SYSTEM (L11.py Style)
# ============================================================================

class RAGSystem:
    """
    RAG system for candidate search
    Similar to L11.py: ChromaDB + Retrieval Chain
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        
        # Get configuration objects
        chromadb_cfg = self.config.get_chromadb_config()
        llm_cfg = self.config.get_llm_config()
        streamlit_cfg = self.config.get_streamlit_config()
        
        self.db_path = Path(chromadb_cfg.db_path)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=chromadb_cfg.collection_name
        )
        
        # Embedding model with config
        embedding_kwargs = {
            "api_key": llm_cfg.api_key,
            "model": llm_cfg.embedding_model
        }
        if llm_cfg.embedding_base_url:
            embedding_kwargs["base_url"] = llm_cfg.embedding_base_url
        
        self.embeddings = OpenAIEmbeddings(**embedding_kwargs)
        
        # LangChain Chroma wrapper (L11.py style)
        self.vectordb = Chroma(
            collection_name=chromadb_cfg.collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client 
        )
        
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={"k": streamlit_cfg.retriever_k}
        )
        
        # Create RAG chain (L11.py style)
        self.rag_chain = self._create_rag_chain()
    
    def _create_rag_chain(self):
        """Create RAG chain (L11.py style)"""
        # Get LLM config
        llm_cfg = self.config.get_llm_config()
        
        # System prompt\
        system_prompt = (
            "You are a helpful assistant for the Millennium BD team. "
            "Use the following retrieved resume information to answer questions about candidates. "
            "\n\n"
            "CRITICAL RULES - YOU MUST FOLLOW THESE:\n"
            "1. ONLY mention candidates whose names and details appear EXPLICITLY in the retrieved context below.\n"
            "2. If the retrieved context is empty, contains no candidate names, or does not match the query, "
            "you MUST respond with: 'No candidates found matching the criteria.'\n"
            "3. NEVER invent, make up, or create candidate names, locations, or details that are not in the context.\n"
            "4. If the context mentions candidates but they don't match the query criteria, explicitly state "
            "'No candidates found matching the criteria' rather than forcing a match.\n"
            "5. Be specific and cite which candidate(s) match the criteria ONLY if their information is clearly present.\n"
            "6. Include candidate names and key details in your response ONLY if they appear in the context.\n"
            "7. If you don't know the answer or the context is insufficient, say: 'No candidates found matching the criteria.'\n"
            "\n"
            "Retrieved context:\n"
            "{context}\n"
            "\n"
            "Remember: The retrieved context above is your ONLY source of information. Do not use any other knowledge."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # LLM with config
        llm_kwargs = {
            "api_key": llm_cfg.api_key,
            "model": llm_cfg.model,
            "temperature": llm_cfg.temperature
        }
        if llm_cfg.base_url:
            llm_kwargs["base_url"] = llm_cfg.base_url
        
        llm = ChatOpenAI(**llm_kwargs)
        
        # Create RAG chain (L11.py style)
        rag_chain = create_retrieval_chain(self.retriever, prompt | llm)
        
        return rag_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with context validation"""
        try:
            response = self.rag_chain.invoke({"input": question})
            context = response.get('context', [])
            answer = response.get('answer', '')
            
            # Extract content if answer is an AIMessage object (create_retrieval_chain returns AIMessage)
            if hasattr(answer, 'content'):
                answer = answer.content
            elif not isinstance(answer, str):
                answer = str(answer)
            
            # Validate context quality
            if not context or len(context) == 0:
                return {
                    'answer': 'No candidates found matching the criteria.',
                    'context': []
                }
            
            # Check if context contains actual candidate information
            # Context from create_retrieval_chain returns Document objects, not dicts
            context_text = ' '.join([str(c.page_content) if isinstance(c, Document) else str(c) for c in context])
            # Look for candidate names in metadata or content
            has_candidate_info = False
            if isinstance(context, list):
                for item in context:
                    # Handle Document objects from LangChain
                    if isinstance(item, Document):
                        # Check metadata for name
                        metadata = item.metadata if hasattr(item, 'metadata') else {}
                        if metadata.get('name') and metadata.get('name') != 'Unknown':
                            has_candidate_info = True
                            break
                        # Check document content for candidate indicators
                        content = item.page_content if hasattr(item, 'page_content') else str(item)
                        if len(content.strip()) > 50:  # Meaningful content
                            has_candidate_info = True
                            break
                    elif isinstance(item, dict):
                        # Fallback for dict format (shouldn't happen but handle it)
                        if item.get('metadata', {}).get('name') and item.get('metadata', {}).get('name') != 'Unknown':
                            has_candidate_info = True
                            break
                        content = str(item.get('page_content', ''))
                        if len(content.strip()) > 50:  # Meaningful content
                            has_candidate_info = True
                            break
                    else:
                        # Handle other types - just check if it has meaningful content
                        content = str(item)
                        if len(content.strip()) > 50:
                            has_candidate_info = True
                            break
            
            # If context is empty or irrelevant, return explicit "no results"
            if not has_candidate_info or len(context_text.strip()) < 50:
                return {
                    'answer': 'No candidates found matching the criteria.',
                    'context': []
                }
            
            return {
                'answer': answer,
                'context': context
            }
        except Exception as e:
            return {
                'answer': f"Error querying: {str(e)}",
                'context': []
            }
    
    def get_all_resumes_metadata(self) -> pd.DataFrame:
        """Get all resume metadata for analytics"""
        try:
            all_results = self.collection.get(include=['metadatas'])
            
            # Extract unique resumes by merging metadata from all chunks
            resumes_dict = {}
            for meta in all_results['metadatas']:
                resume_uuid = meta.get('resume_uuid')
                if not resume_uuid:
                    continue
                
                if resume_uuid not in resumes_dict:
                    resumes_dict[resume_uuid] = {}
                
                # Merge metadata, prioritizing non-null/non-empty values
                for key, value in meta.items():
                    if key == 'resume_uuid':
                        continue
                    # Only update if current value is None/empty and new value is not
                    if key not in resumes_dict[resume_uuid] or not resumes_dict[resume_uuid][key]:
                        if value is not None and value != '':
                            resumes_dict[resume_uuid][key] = value
                    # Or if new value is better (non-empty vs empty)
                    elif value is not None and value != '' and (resumes_dict[resume_uuid][key] is None or resumes_dict[resume_uuid][key] == ''):
                        resumes_dict[resume_uuid][key] = value
            
            if resumes_dict:
                # Convert to list and ensure required fields exist
                metadata_list = []
                for resume_uuid, meta in resumes_dict.items():
                    meta['resume_uuid'] = resume_uuid
                    # Ensure experience_years exists and is numeric
                    if 'experience_years' not in meta or meta['experience_years'] is None or meta['experience_years'] == '':
                        meta['experience_years'] = 0.0
                    else:
                        try:
                            meta['experience_years'] = float(meta['experience_years'])
                        except (ValueError, TypeError):
                            meta['experience_years'] = 0.0
                    metadata_list.append(meta)
                
                df = pd.DataFrame(metadata_list)
                df = df.dropna(subset=['name'])
                
                # Ensure experience_years column exists even if empty
                if 'experience_years' not in df.columns:
                    df['experience_years'] = 0.0
                
                return df
            else:
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=['resume_uuid', 'name', 'experience_years', 'geography', 'sectors', 'skills', 'investment_approach'])
        except Exception as e:
            st.error(f"Error retrieving metadata: {e}")
            return pd.DataFrame(columns=['resume_uuid', 'name', 'experience_years', 'geography', 'sectors', 'skills', 'investment_approach'])
    
    def get_resume_count(self) -> int:
        """Get total number of unique resumes"""
        try:
            all_results = self.collection.get()
            unique_ids = set()
            for meta in all_results.get('metadatas', []):
                resume_uuid = meta.get('resume_uuid')
                if resume_uuid:
                    unique_ids.add(resume_uuid)
            return len(unique_ids)
        except Exception as e:
            return 0

# ============================================================================
# CANDIDATE SUGGESTER
# ============================================================================

class CandidateSuggester:
    """
    Scores and suggests candidates based on job requisition criteria
    """
    
    def __init__(self, rag_system: RAGSystem, config: Optional[AppConfig] = None):
        self.rag_system = rag_system
        self.config = config or get_config()
    
    def suggest_candidates(
        self,
        geography: Optional[str] = None,
        min_experience: float = 0.0,
        sectors: Optional[List[str]] = None,
        investment_approach: Optional[str] = None,
        skills: Optional[List[str]] = None,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest top candidates based on criteria
        """
        # Use config default if top_n not provided
        if top_n is None:
            streamlit_cfg = self.config.get_streamlit_config()
            top_n = streamlit_cfg.suggestion_top_n
        
        # Get all resume metadata
        df = self.rag_system.get_all_resumes_metadata()
        
        if df.empty:
            return []
        
        # Score candidates
        scored_candidates = []
        
        for _, row in df.iterrows():
            score = 0.0
            reasons = []
            
            # Geography match (weight: 2.0)
            if geography:
                if row.get('geography') == geography:
                    score += 2.0
                    reasons.append(f"Geography match: {geography}")
                else:
                    reasons.append(f"Geography mismatch: {row.get('geography')} vs {geography}")
            
            # Experience match (weight: 1.5) - only score if requirement specified
            try:
                exp_years = float(row.get('experience_years', 0))
            except (ValueError, TypeError):
                exp_years = 0.0
            
            if min_experience > 0:  # Only award points when requirement is specified
                if exp_years >= min_experience:
                    score += 1.5
                    reasons.append(f"Experience: {exp_years:.1f} years (required: {min_experience:.1f})")
                else:
                    reasons.append(f"Experience insufficient: {exp_years:.1f} years (required: {min_experience:.1f})")
            else:
                reasons.append(f"Experience: {exp_years:.1f} years (no requirement)")
            
            # Investment approach match (weight: 2.0)
            if investment_approach:
                if row.get('investment_approach') == investment_approach:
                    score += 2.0
                    reasons.append(f"Investment approach match: {investment_approach}")
                else:
                    reasons.append(f"Investment approach mismatch")
            
            # Sector match (weight: 1.5 per match) - improved matching
            if sectors:
                resume_sectors_str = str(row.get('sectors', '')).lower()
                resume_sectors = [s.strip().lower() for s in resume_sectors_str.split(',') if s.strip()]
                matched_sectors = []
                for req_sector in sectors:
                    req_sector_lower = req_sector.strip().lower()
                    # Check exact match or substring match
                    if req_sector_lower in resume_sectors or any(req_sector_lower in rs for rs in resume_sectors):
                        matched_sectors.append(req_sector)
                
                if matched_sectors:
                    score += 1.5 * len(matched_sectors)
                    reasons.append(f"Sector match: {', '.join(matched_sectors)}")
                else:
                    reasons.append(f"No sector match (resume: {resume_sectors_str[:50]})")
            
            # Skills match (weight: 0.5 per skill)
            if skills:
                resume_skills = str(row.get('skills', '')).lower()
                matched_skills = [s for s in skills if s.lower() in resume_skills]
                if matched_skills:
                    score += 0.5 * len(matched_skills)
                    reasons.append(f"Skills match: {', '.join(matched_skills)}")
                else:
                    reasons.append(f"No skills match")
            
            scored_candidates.append({
                'resume_uuid': row.get('resume_uuid'),
                'name': row.get('name', 'Unknown'),
                'geography': row.get('geography'),
                'experience_years': exp_years,
                'investment_approach': row.get('investment_approach'),
                'sectors': row.get('sectors'),
                'score': score,
                'reasons': reasons
            })
        
        # Sort by score (descending), then by experience_years (descending) as tie-breaker
        scored_candidates.sort(key=lambda x: (x['score'], x['experience_years']), reverse=True)
        
        # Filter out 0-score candidates when criteria are provided
        has_criteria = any([geography, min_experience > 0, sectors, investment_approach, skills])
        if has_criteria:
            scored_candidates = [c for c in scored_candidates if c['score'] > 0]
        
        return scored_candidates[:top_n]

# ============================================================================
# QUERY AGENT (LangChain Agent)
# ============================================================================

class QueryAgent:
    """
    Intelligent agent that understands user queries and filters accordingly
    Reactive agent that interprets requirements
    """
    
    def __init__(self, rag_system: RAGSystem, suggester: CandidateSuggester, config: Optional[AppConfig] = None):
        self.rag_system = rag_system
        self.suggester = suggester
        self.config = config or get_config()
        
        # Create agent tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        return [
            Tool(
                name="search_candidates",
                func=self._search_candidates,
                description="Search candidates by natural language query. Use this for general questions about candidates, their experience, skills, or background."
            ),
            Tool(
                name="suggest_candidates",
                func=self._suggest_candidates_tool,
                description="Suggest top candidates based on job requisition criteria: geography, experience, sectors, investment approach, skills. Returns ranked list with scores."
            ),
            Tool(
                name="get_candidate_details",
                func=self._get_candidate_details,
                description="Get detailed information about a specific candidate by name or resume UUID."
            ),
        ]
    
    def _create_agent(self):
        """Create OpenAI functions agent"""
        llm_cfg = self.config.get_llm_config()
        
        llm_kwargs = {
            "api_key": llm_cfg.api_key,
            "model": llm_cfg.model,
            "temperature": llm_cfg.temperature
        }
        if llm_cfg.base_url:
            llm_kwargs["base_url"] = llm_cfg.base_url
        
        llm = ChatOpenAI(**llm_kwargs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent assistant for the Millennium BD team.
            Your job is to help users find the right candidates based on their requirements.
            
            CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
            1. NEVER make up or invent candidate names, companies, locations, certifications, or any details
            2. ONLY use candidates returned by the tools (search_candidates, suggest_candidates, get_candidate_details)
            3. If tools return no results or empty results, you MUST explicitly say "No candidates found matching the criteria" - do NOT invent candidates
            4. If you don't have information from tools, say "No candidates found matching the criteria" - do NOT make up information
            5. NEVER create fictional candidate profiles like "Sarah Johnson" or "Michael Chen" unless they appear in tool results
            6. NEVER infer candidate details that are not explicitly in the tool results
            7. If tool results say "No candidates found", DO NOT try to be helpful by creating examples
            
            WHAT NOT TO DO (EXAMPLES OF FORBIDDEN BEHAVIOR):
            WRONG: "I found 2 candidates: Sarah Johnson (London) and Michael Chen (Frankfurt)" - if these names don't appear in tool results
            WRONG: "Based on the search, here are some candidates..." - if search returned no results
            WRONG: Creating candidate profiles from general knowledge or assumptions
            WRONG: Saying "I'll search for candidates..." and then inventing results without actually using tools
            
            CORRECT: "No candidates found matching the criteria" - if tools return no results
            CORRECT: Using ONLY candidate names and details that appear in tool results
            CORRECT: If tool returns "No candidates found", you say "No candidates found matching the criteria"
            
            When a user asks a question:
            1. Understand what they're looking for (geography, experience, skills, sectors, etc.)
            2. Use the appropriate tool:
               - "search_candidates" for general questions about candidates
               - "suggest_candidates" when they provide job requisition criteria
               - "get_candidate_details" for specific candidate information
            3. Check the tool results carefully:
               - If results are empty or say "No candidates found", respond with "No candidates found matching the criteria"
               - If results contain candidate information, use ONLY that information
            4. Use ONLY the information returned by the tools - nothing else
            5. Provide clear, helpful answers with candidate names and details FROM THE TOOL RESULTS ONLY
            6. When suggesting candidates, explain why each candidate matches based on tool results
            
            REMEMBER: Tool results are your ONLY source of truth. If tools don't return candidate information, 
            you MUST say "No candidates found matching the criteria" - do NOT invent or create candidates.
            
            IMPORTANT OUTPUT FORMATTING RULES:
            - NEVER include XML tags, function call syntax, or technical implementation details in your responses
            - NEVER write function calls like <invoke name="..."> or <function_calls> in your text
            - NEVER include function names like "search_candidates(" or "suggest_candidates(" in your natural language responses
            - Provide ONLY natural, conversational text responses to users
            - Function calls are handled automatically by the system - you don't need to show them
            - If you need to use a tool, the system will call it automatically - just respond naturally to the user"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=False
        )
    
    def _search_candidates(self, query: str) -> str:
        """
        Search candidates using RAG system.
        
        Args:
            query: Natural language search query
            
        Returns:
            Answer string with candidate information or "No candidates found" message
        """
        result = self.rag_system.query(query)
        answer = result.get('answer', '')
        context = result.get('context', [])
        
        # Ensure answer is a string (should already be handled by RAGSystem.query, but double-check)
        if hasattr(answer, 'content'):
            answer = answer.content
        elif not isinstance(answer, str):
            answer = str(answer)
        
        logger.debug(f"Search query: {query}, found {len(context)} context documents")
        
        # If no context found, make it clear
        if not context:
            return "No candidates found in the database matching this query. Please try different search terms or criteria."
        
        return answer
    
    def _suggest_candidates_tool(self, criteria: str) -> str:
        """Suggest candidates based on criteria"""
        # Parse criteria from natural language (simplified)
        geography = None
        min_experience = 0.0
        sectors = []
        investment_approach = None
        
        # Simple keyword extraction (can be enhanced with LLM)
        criteria_lower = criteria.lower()
        if 'us' in criteria_lower or 'united states' in criteria_lower:
            geography = 'US'
        elif 'europe' in criteria_lower:
            geography = 'Europe'
        elif 'asia' in criteria_lower:
            geography = 'Asia-Pacific'
        
        if 'fundamental' in criteria_lower:
            investment_approach = 'Fundamental'
        elif 'systematic' in criteria_lower or 'quantitative' in criteria_lower:
            investment_approach = 'Systematic/Quantitative'
        
        # Extract experience (simple pattern matching)
        exp_match = re.search(r'(\d+)\+?\s*years?', criteria_lower)
        if exp_match:
            min_experience = float(exp_match.group(1))
        
        logger.debug(
            f"Suggest candidates criteria - geography: {geography}, "
            f"min_experience: {min_experience}, sectors: {sectors}, "
            f"approach: {investment_approach}"
        )
        
        # Get suggestions
        suggestions = self.suggester.suggest_candidates(
            geography=geography,
            min_experience=min_experience,
            sectors=sectors,
            investment_approach=investment_approach,
            top_n=5
        )
        
        logger.debug(f"Found {len(suggestions)} candidate suggestions")
        
        if not suggestions:
            return "No candidates found matching the criteria. The database does not contain any candidates matching these requirements. Please try different criteria."
        
        # Format response clearly for agent
        response = f"Found {len(suggestions)} candidate(s) matching the criteria:\n\n"
        for i, candidate in enumerate(suggestions, 1):
            response += f"{i}. {candidate['name']})\n"
            response += f"   Geography: {candidate['geography']}\n"
            response += f"   Experience: {candidate['experience_years']:.1f} years\n"
            if candidate.get('investment_approach'):
                response += f"   Investment Approach: {candidate['investment_approach']}\n"
            if candidate.get('reasons'):
                response += f"   Match Reasons: {'; '.join(candidate['reasons'][:3])}\n"
            response += "\n"
        
        response += "\nIMPORTANT: Only use these candidates in your response. Do not make up or invent any other candidate names."
        
        return response
    
    def _get_candidate_details(self, candidate_name: str) -> str:
        """Get details about a specific candidate"""
        query = f"Tell me everything about candidate {candidate_name}"
        result = self.rag_system.query(query)
        answer = result.get('answer', '')
        
        # Ensure answer is a string (should already be handled by RAGSystem.query, but double-check)
        if hasattr(answer, 'content'):
            answer = answer.content
        elif not isinstance(answer, str):
            answer = str(answer)
        
        return answer
    
    def process_query(self, user_query: str, chat_history: Optional[List] = None) -> str:
        """Process user query through agent"""
        try:
            # Convert chat history to LangChain message format
            if chat_history is None:
                langchain_history = []
            else:
                langchain_history = []
                for msg in chat_history:
                    if msg.get("role") == "user":
                        langchain_history.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        langchain_history.append(AIMessage(content=msg.get("content", "")))
            
            logger.debug(f"Processing query: {user_query}")
            
            # Invoke agent with error handling
            try:
                result = self.agent.invoke({
                    "input": user_query,
                    "chat_history": langchain_history
                })
            except Exception as agent_error:
                logger.error(f"Agent execution error: {str(agent_error)}", exc_info=True)
                # Check if it's a parsing error
                if "parsing" in str(agent_error).lower() or "function" in str(agent_error).lower():
                    return "I encountered an error while processing your query. Please try rephrasing your question or be more specific about what you're looking for."
                else:
                    return f"Error processing query: {str(agent_error)}. Please try again."
            
            output = result.get("output", "")
            
            # Extract content if output is an AIMessage object
            if hasattr(output, 'content'):
                output = output.content
            elif not isinstance(output, str):
                output = str(output)
            
            logger.debug(f"Agent output length: {len(output)}")
            
            # Validation: Check if output contains common hallucinated names
            hallucinated_names = ["Sarah Chen", "Michael Rodriguez", "Jennifer Park", "John Smith", "Jane Doe"]
            if any(name in output for name in hallucinated_names):
                # Check if these names actually came from tools by checking if they're in database
                df = self.rag_system.get_all_resumes_metadata()
                actual_names = set(df['name'].str.lower() if 'name' in df.columns else [])
                for name in hallucinated_names:
                    if name in output and name.lower() not in actual_names:
                        logger.warning(f"Output contains potentially hallucinated name: {name}")
                        return f"Error: The system attempted to return candidate information that doesn't exist in the database. Please try rephrasing your query or use the search filters to find real candidates."
            
            # Final validation: ensure output is not empty
            if not output or len(output.strip()) < 1:
                return "I couldn't generate a response to your query. Please try rephrasing your question or use the search filters to find candidates."
            
            return output
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}. Please try again or rephrase your question."

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Set page config ONCE at module level
config = get_config()
streamlit_cfg = config.get_streamlit_config()
st.set_page_config(
    page_title=streamlit_cfg.page_title,
    page_icon=streamlit_cfg.page_icon,
    layout=streamlit_cfg.layout
)

def main():
    """Main Streamlit app"""
    # Get configuration
    config = st.session_state.get('config') or get_config()
    if 'config' not in st.session_state:
        st.session_state.config = config
    
    st.title("Millennium Resume Search Platform")
    st.markdown("** Vision-Powered Resume Parser with LangChain Agent**")
    st.markdown("---")
    
    # Initialize components with config
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = RAGSystem(config=config)
            st.session_state.suggester = CandidateSuggester(
                st.session_state.rag_system,
                config=config
            )
            st.session_state.agent = QueryAgent(
                st.session_state.rag_system,
                st.session_state.suggester,
                config=config
            )
    
    # Add parser to session state to avoid re-initialization
    if 'parser' not in st.session_state:
        with st.spinner("Initializing parser..."):
            try:
                st.session_state.parser = ResumeParser(config=config)
            except Exception as e:
                st.error(f"Parser initialization failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                st.session_state.parser = None
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Agent Search", "Analytics", "Upload & Parse"])
    
    with tab1:
        st.header("Ask the Agent")
        st.markdown("""
        **Examples:**
        - "Find candidates with 3+ years experience in quantitative trading"
        - "Show me all candidates from Europe with CFA certification"
        - "Who has experience in Technology sector with Python skills?"
        - "Suggest candidates for a Fundamental equity analyst role in US"
        """)
        
        # Job Requisition Form
        with st.expander("Job Requisition Form", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                req_geography = st.selectbox(
                    "Geography",
                    options=[None, "US", "Europe", "Asia-Pacific", "Other"],
                    format_func=lambda x: "All" if x is None else x
                )
                
                req_investment_approach = st.selectbox(
                    "Investment Approach",
                    options=[None, "Fundamental", "Systematic/Quantitative", "Mixed"],
                    format_func=lambda x: "All" if x is None else x
                )
            
            with col2:
                req_min_experience = st.slider(
                    "Minimum Experience (years)",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.5
                )
                
                req_sectors = st.multiselect(
                    "Sectors",
                    options=["Technology", "Healthcare", "Financial Services", 
                            "Energy", "Industrials", "Consumer", "Credit", "Macro"],
                    default=[]
                )
            
            if st.button("Get Suggestions", type="primary"):
                with st.spinner("Finding best candidates..."):
                    suggestions = st.session_state.suggester.suggest_candidates(
                        geography=req_geography,
                        min_experience=req_min_experience,
                        sectors=req_sectors,
                        investment_approach=req_investment_approach,
                        top_n=5
                    )
                    
                    if suggestions:
                        st.success(f"Found {len(suggestions)} top candidates")
                        
                        for i, candidate in enumerate(suggestions, 1):
                            with st.expander(
                                f"**{i}. {candidate['name']}**",
                                expanded=(i == 1)
                            ):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Geography:** {candidate['geography']}")
                                    st.write(f"**Experience:** {candidate['experience_years']:.1f} years")
                                    st.write(f"**Investment Approach:** {candidate['investment_approach']}")
                                    st.write(f"**Resume UUID:** `{candidate['resume_uuid']}`")
                                
                                with col2:
                                    st.write("**Match Reasons:**")
                                    for reason in candidate['reasons']:
                                        st.write(f"- {reason}")
                    else:
                        st.info("No candidates found matching the criteria")
        
        # Chat Interface
        st.markdown("---")
        st.subheader("Natural Language Search")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask about candidates..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Pass chat history (excluding the current message we just added)
                    chat_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
                    response = st.session_state.agent.process_query(prompt, chat_history=chat_history)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with tab2:
        st.header("Analytics Dashboard")
        
        # Get metadata
        df = st.session_state.rag_system.get_all_resumes_metadata()
        total_resumes = st.session_state.rag_system.get_resume_count()
        
        if not df.empty:
            # Row 1: Key Metrics
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Resumes", total_resumes)
            
            with col2:
                try:
                    avg_exp = df['experience_years'].mean() if 'experience_years' in df.columns else 0.0
                    st.metric("Avg Experience", f"{avg_exp:.1f} years")
                except Exception:
                    st.metric("Avg Experience", "N/A")
            
            with col3:
                try:
                    if 'geography' in df.columns:
                        unique_geos = df['geography'].nunique()
                        st.metric("Geographies", unique_geos)
                    else:
                        st.metric("Geographies", "N/A")
                except Exception:
                    st.metric("Geographies", "N/A")
            
            with col4:
                try:
                    if 'skills' in df.columns:
                        all_skills = []
                        for skills_str in df['skills'].dropna():
                            if skills_str and str(skills_str).strip():
                                all_skills.extend([s.strip() for s in str(skills_str).split(',') if s.strip()])
                        unique_skills = len(set(all_skills))
                        st.metric("Total Skills", unique_skills)
                    else:
                        st.metric("Total Skills", "N/A")
                except Exception:
                    st.metric("Total Skills", "N/A")
            
            st.markdown("---")
            
            # Row 2: Geography and Investment Approach
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    if 'geography' in df.columns:
                        geo_counts = df['geography'].value_counts()
                        if not geo_counts.empty:
                            fig = px.pie(
                                values=geo_counts.values,
                                names=geo_counts.index,
                                title="Geography Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig.update_layout(height=400, showlegend=True)
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.info("No geography data available")
                    else:
                        st.info("Geography column not found in data")
                except Exception as e:
                    st.error(f"Error displaying geography chart: {e}")
            
            with col2:
                try:
                    if 'investment_approach' in df.columns:
                        approach_counts = df['investment_approach'].value_counts()
                        if not approach_counts.empty:
                            fig = px.bar(
                                x=approach_counts.index,
                                y=approach_counts.values,
                                title="Investment Approach Distribution",
                                labels={'x': 'Investment Approach', 'y': 'Count'},
                                color=approach_counts.values,
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig, width='stretch')
                        # Skip section if no data available
                    # Skip section if column not found
                except Exception as e:
                    st.error(f"Error displaying investment approach chart: {e}")
            
            # Row 3: Experience and Sectors
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    if 'experience_years' in df.columns:
                        exp_data = df['experience_years'].dropna()
                        if not exp_data.empty:
                            fig = px.histogram(
                                df,
                                x='experience_years',
                                nbins=20,
                                title="Experience Distribution (Years)",
                                labels={'experience_years': 'Years of Experience', 'count': 'Number of Candidates'},
                                color_discrete_sequence=['#2E86AB']
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.info("No experience data available")
                    else:
                        st.info("Experience years column not found in data")
                except Exception as e:
                    st.error(f"Error displaying experience chart: {e}")
            
            with col2:
                try:
                    if 'sectors' in df.columns:
                        all_sectors = []
                        for sectors_str in df['sectors'].dropna():
                            if sectors_str and str(sectors_str).strip():
                                all_sectors.extend([s.strip() for s in str(sectors_str).split(',') if s.strip()])
                        
                        if all_sectors:
                            sector_counts = pd.Series(all_sectors).value_counts().head(10)
                            if not sector_counts.empty:
                                fig = px.bar(
                                    x=sector_counts.values,
                                    y=sector_counts.index,
                                    orientation='h',
                                    title="Top Sectors",
                                    labels={'x': 'Count', 'y': 'Sector'},
                                    color=sector_counts.values,
                                    color_continuous_scale='Viridis'
                                )
                                fig.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig, width='stretch')
                            # Skip section if no sectors data available
                        # Skip section if no sectors found
                    # Skip section if column not found
                except Exception as e:
                    st.error(f"Error displaying sectors chart: {e}")
            
            st.markdown("---")
            
            # Row 4: Data Summary Table
            st.subheader("Data Summary")
            try:
                summary_data = []
                for _, row in df.iterrows():
                    summary_data.append({
                        'Name': row.get('name', 'Unknown'),
                        'Geography': row.get('geography', 'N/A'),
                        'Experience (Years)': f"{row.get('experience_years', 0):.1f}",
                        'Sectors': row.get('sectors', 'N/A')[:50] + '...' if len(str(row.get('sectors', ''))) > 50 else row.get('sectors', 'N/A'),
                        'Skills Count': len(str(row.get('skills', '')).split(',')) if row.get('skills') else 0
                    })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, width='stretch', hide_index=True)
                else:
                    st.info("No summary data available")
            except Exception as e:
                    st.error(f"Error displaying summary table: {e}")
        else:
            st.info("No data available. Please parse resumes first using the 'Upload & Parse' tab.")
    
    with tab3:
        st.header("Upload & Parse Resumes")
        st.markdown("**Upload resumes to parse and extract structured data using vision API**")
        st.caption("Documents are validated to ensure they are resumes before processing. vision extracts text with better accuracy from scanned PDFs and complex layouts.")
        
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF or DOCX)",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            key='resume_file_uploader'
        )
        
        if uploaded_files:
            # Initialize cache in session state
            if 'parsed_files_cache' not in st.session_state:
                st.session_state.parsed_files_cache = {}
            
            # Use config from session state
            config = st.session_state.get('config') or get_config()
            parser = st.session_state.get('parser')
            if parser is None:
                st.error("Parser not initialized. Please refresh the page.")
                st.stop()
            db_store = ChromaDBStore(config=config)
            
            for file in uploaded_files:
                st.markdown("---")
                st.subheader(f"{file.name}")
                
                # Calculate file hash for caching
                file_bytes = file.getbuffer()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                
                # Check if file is already in cache
                cached_data = st.session_state.parsed_files_cache.get(file_hash)
                
                # Save temporarily
                import tempfile
                # Extract file extension from uploaded file name
                file_ext = Path(file.name).suffix
                if not file_ext:
                    raise ValueError(f"Cannot determine file extension for file: {file.name}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(file_bytes)
                
                try:
                    # Create tabs for this file
                    parse_tab, structured_tab, chunks_tab = st.tabs([
                        "Parsed Data", 
                        "Structured Output",
                        "Document Chunks"
                    ])
                    
                    # Process file if not cached
                    if cached_data is None:
                        # Step 1: Validate document (if validation enabled)
                        validation_result = None
                        validation_cfg = config.get_validation_config()
                        if validation_cfg.enable_resume_validation:
                            from document_validator import DocumentValidator
                            validator = DocumentValidator(config=config)
                            
                            with st.spinner("Validating document..."):
                                try:
                                    validation_result = validator.validate_resume_preview(Path(tmp_path))
                                    
                                    # Display validation result
                                    if validation_result.is_resume:
                                        st.success(
                                            f"Document validated as resume "
                                            f"(Confidence: {validation_result.confidence:.1%})"
                                        )
                                        if validation_result.reason:
                                            st.caption(f"Reason: {validation_result.reason}")
                                    else:
                                        st.error(
                                            f"Document validation failed: {validation_result.reason}"
                                        )
                                        st.warning(
                                            f"Confidence: {validation_result.confidence:.1%}. "
                                            f"Detected document type: {validation_result.document_type or 'unknown'}"
                                        )
                                        st.info(
                                            "Please upload a valid resume/CV document. "
                                            "A resume typically contains: name, contact info, work experience, education, and skills."
                                        )
                                        # Cleanup and skip processing
                                        import os
                                        if os.path.exists(tmp_path):
                                            os.unlink(tmp_path)
                                        continue
                                except Exception as e:
                                    st.warning(f"Validation check failed: {e}. Proceeding with parsing...")
                                    logger.warning(f"Validation check failed: {e}", exc_info=True)
                        
                        # Step 2: Parse resume
                        with st.spinner("Parsing document."):
                            try:
                                # Parse resume
                                parsed = parser.parse_resume(tmp_path)
                                
                                # Store in ChromaDB
                                db_store.store_resume(parsed)
                                
                                # Cache the results
                                st.session_state.parsed_files_cache[file_hash] = {
                                    'parsed': parsed,
                                    'file_name': file.name,
                                    'validation_result': validation_result
                                }
                                cached_data = st.session_state.parsed_files_cache[file_hash]
                            except ValueError as e:
                                # Handle validation errors
                                if "Document validation failed" in str(e):
                                    st.error(f"❌ {str(e)}")
                                    import os
                                    if os.path.exists(tmp_path):
                                        os.unlink(tmp_path)
                                    continue
                                else:
                                    raise
                    else:
                        st.info("📋 Using cached results (file already processed)")
                        parsed = cached_data['parsed']
                        validation_result = cached_data.get('validation_result')
                    
                    with parse_tab:
                        if cached_data is None:
                            st.success(f"Successfully parsed and stored {file.name}")
                        else:
                            st.info(f"Displaying cached results for {file.name}")
                        
                        # Display validation result if available
                        if validation_result:
                            with st.expander("Validation Details", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Is Resume", "Yes" if validation_result.is_resume else "No")
                                    st.metric("Confidence", f"{validation_result.confidence:.1%}")
                                with col2:
                                    st.write("**Reason:**")
                                    st.write(validation_result.reason)
                                    if validation_result.document_type:
                                        st.write(f"**Document Type:** {validation_result.document_type}")
                        
                        # Display key information
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Basic Information")
                            st.json({
                                'resume_uuid': parsed['resume_uuid'],
                                'name': parsed['structured_data'].name,
                                'email': parsed['structured_data'].email,
                                'phone': parsed['structured_data'].phone,
                                'location': parsed['structured_data'].location,
                                'geography': parsed['structured_data'].geography.value,
                                'experience_years': parsed['structured_data'].experience_years
                            })
                        
                        with col2:
                            st.markdown("#### Summary")
                            if parsed['structured_data'].summary:
                                st.text_area(
                                    "Professional Summary",
                                    parsed['structured_data'].summary,
                                    height=150,
                                    disabled=True
                                )
                        
                        # Work Experience
                        if parsed['structured_data'].work_experience:
                            st.markdown("#### Work Experience")
                            exp_data = []
                            for exp in parsed['structured_data'].work_experience:
                                exp_data.append({
                                    'Company': exp.company,
                                    'Position': exp.position,
                                    'Duration': f"{exp.start_date} - {exp.end_date or 'Present'}",
                                    'Sector': exp.sector or 'N/A'
                                })
                            st.dataframe(pd.DataFrame(exp_data), width='stretch', hide_index=True)
                        
                        # Education
                        if parsed['structured_data'].education:
                            st.markdown("#### Education")
                            edu_data = []
                            for edu in parsed['structured_data'].education:
                                edu_data.append({
                                    'Degree': edu.degree,
                                    'Field': edu.field,
                                    'Institution': edu.institution,
                                    'Year': edu.graduation_year or 'N/A'
                                })
                            st.dataframe(pd.DataFrame(edu_data), width='stretch', hide_index=True)
                        
                        # Skills
                        if parsed['structured_data'].skills:
                            st.markdown("#### Skills")
                            skills_str = ", ".join(parsed['structured_data'].skills)
                            st.text(skills_str)
                    
                    with structured_tab:
                        st.markdown("### Complete Structured Output")
                        st.markdown("**Full JSON representation of extracted data**")
                        
                        # Convert to JSON-serializable format
                        structured_dict = parsed['structured_data'].model_dump()
                        
                        # Add extraction diagnostics
                        st.markdown("#### Extraction Diagnostics")
                        structured = parsed['structured_data']
                        
                        # Check each field and show status
                        diagnostics = []
                        
                        # Basic info
                        if not structured.name or structured.name == 'Unknown':
                            diagnostics.append(("Warning", "Name", "Missing or not extracted"))
                        else:
                            diagnostics.append(("Success", "Name", f"Extracted: {structured.name}"))
                        
                        if not structured.email:
                            diagnostics.append(("Warning", "Email", "Not found"))
                        else:
                            diagnostics.append(("Success", "Email", f"Extracted: {structured.email}"))
                        
                        if not structured.location or structured.location == 'Unknown':
                            diagnostics.append(("Warning", "Location", "Missing or not extracted"))
                        else:
                            diagnostics.append(("Success", "Location", f"Extracted: {structured.location}"))
                        
                        # Work experience
                        if not structured.work_experience:
                            diagnostics.append(("Warning", "Work Experience", f"0 entries extracted (check chunks for work-related content)"))
                        else:
                            diagnostics.append(("Success", "Work Experience", f"{len(structured.work_experience)} entries extracted"))
                        
                        # Education
                        if not structured.education:
                            diagnostics.append(("Warning", "Education", f"0 entries extracted (check chunks for education-related content)"))
                        else:
                            diagnostics.append(("Success", "Education", f"{len(structured.education)} entries extracted"))
                        
                        # Skills
                        skills_count = len(structured.skills) + len(structured.programming_languages) + len(structured.tools)
                        if skills_count == 0:
                            diagnostics.append(("Warning", "Skills", f"0 skills extracted (check chunks for skill-related content)"))
                        else:
                            diagnostics.append(("Success", "Skills", f"{len(structured.skills)} skills, {len(structured.programming_languages)} languages, {len(structured.tools)} tools"))
                        
                        # Certifications
                        if not structured.certifications:
                            diagnostics.append(("Info", "Certifications", "0 entries (may not be present in resume)"))
                        else:
                            diagnostics.append(("Success", "Certifications", f"{len(structured.certifications)} entries extracted"))
                        
                        # Display diagnostics in columns
                        col1, col2, col3 = st.columns(3)
                        for i, (status_type, field, status) in enumerate(diagnostics):
                            col = [col1, col2, col3][i % 3]
                            with col:
                                st.markdown(f"**{field}**")
                                st.caption(status)
                        
                        # Show warnings if critical fields are missing
                        warnings = []
                        if not structured.name or structured.name == 'Unknown':
                            warnings.append("**Name** is missing - this is a critical field")
                        if not structured.work_experience:
                            warnings.append("**Work Experience** is empty - check if resume contains work history")
                        if not structured.education:
                            warnings.append("**Education** is empty - check if resume contains education section")
                        if skills_count == 0:
                            warnings.append("**Skills** are empty - check if resume contains skills section")
                        
                        if warnings:
                            st.markdown("---")
                            st.warning("**Extraction Warnings:**")
                            for warning in warnings:
                                st.markdown(f"- {warning}")
                            st.info("If fields are empty but chunks contain relevant content, try re-uploading the file or check if the resume format is supported.")
                        
                        st.markdown("---")
                        st.json(structured_dict)
                        
                        # Download button
                        json_str = json.dumps(structured_dict, indent=2, default=str)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"{file.name}_structured.json",
                            mime="application/json"
                        )
                    
                    with chunks_tab:
                        st.markdown("### Document Chunks")
                        chunks = parsed.get('chunks', [])
                        st.markdown(f"**{len(chunks)} chunks created from document**")
                        
                        # Show chunk summary
                        section_counts = {}
                        for chunk in chunks:
                            section = chunk.get('section', 'unknown')
                            section_counts[section] = section_counts.get(section, 0) + 1
                        
                        if section_counts:
                            st.markdown("#### Chunk Summary by Section")
                            summary_df = pd.DataFrame([
                                {'Section': section, 'Count': count}
                                for section, count in section_counts.items()
                            ])
                            st.dataframe(summary_df, hide_index=True, width='stretch')
                        
                        # Compare chunks with structured data
                        st.markdown("#### Chunk vs Structured Data Comparison")
                        structured = parsed['structured_data']
                        
                        # Check if chunks contain content that should be in structured data
                        
    
                        
                        st.markdown("---")
                        
                        for idx, chunk in enumerate(chunks):
                            with st.expander(f"Chunk {idx + 1}: {chunk.get('section', 'unknown')}", expanded=False):
                                st.markdown(f"**Section:** {chunk.get('section', 'unknown')}")
                                st.markdown(f"**Chunk UUID:** `{chunk.get('chunk_uuid', 'N/A')}`")
                                if 'page' in chunk.get('metadata', {}):
                                    st.markdown(f"**Page:** {chunk['metadata']['page']}")
                                st.markdown("**Content:**")
                                content = chunk.get('content', '')
                                if len(content) > 1000:
                                    st.code(content[:1000] + "\n\n... (truncated)", language='text')
                                    st.caption(f"Full content length: {len(content)} characters")
                                else:
                                    st.code(content, language='text')
                    
                    # Cleanup temp file
                    import os
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                    logger.error(f"Error processing {file.name}: {e}", exc_info=True)
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
            
            st.success("All files processed! Refresh the page to see new resumes in search results.")

if __name__ == "__main__":
    main()
