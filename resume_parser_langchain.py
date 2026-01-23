#!/usr/bin/env python
# coding: utf-8

"""
LangChain-based Resume Parser

This module provides a comprehensive resume parsing system using LangChain and vision API.
It extracts structured data from PDF and DOCX resume files and stores them in ChromaDB for search.

Author: Siddharth Singh
"""

import json
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import Chroma

import chromadb
from chromadb.config import Settings

# Import configuration
from config import get_config, AppConfig

# Import vision loader and validator
from vision_document_loader import VisionDocumentLoader
from document_validator import DocumentValidator

# Import utility functions
from resume_utils import generic_resume_extraction, update_task_log_only

# Configure logging
logger = logging.getLogger(__name__)




class Geography(str, Enum):
    US = "US"
    EUROPE = "Europe"
    ASIA_PACIFIC = "Asia-Pacific"
    OTHER = "Other"

class WorkExperience(BaseModel):
    """Detailed work experience schema"""
    company: str = Field(default="", description="Company name - extract the full company name exactly as written, return empty string if not present")
    position: str = Field(default="", description="Job title/position - extract the exact job title including prefixes like 'Senior', 'Lead', etc., return empty string if not present")
    start_date: str = Field(default="", description="Start date - extract in any format found (month/year, year only, or full date), return empty string if not present")
    end_date: str = Field(default="", description="End date or 'Present' if currently employed - return empty string if not present")
    duration_months: Optional[float] = Field(default=None, description="Duration in months - calculate from start and end dates if possible")
    location: str = Field(default="", description="Job location - city/country where the job was located, return empty string if not mentioned")
    description: str = Field(default="", description="Job responsibilities and achievements - extract ALL bullet points, responsibilities, and accomplishments comprehensively, return empty string if not present")
    sector: str = Field(default="", description="Industry sector - identify sector like 'Finance', 'Technology', 'Healthcare', etc., infer from company name or description if needed, return empty string if not present")

class Education(BaseModel):
    """Education schema"""
    degree: str = Field(default="", description="Degree - extract full degree name (Bachelor's, Master's, MBA, PhD, etc.) or abbreviations (BS, BA, MS, MA, MBA, PhD) as written, return empty string if not present")
    field: str = Field(default="", description="Field of study - extract major, specialization, or field (e.g., Computer Science, Business Administration, Engineering), return empty string if not present")
    institution: str = Field(default="", description="University/Institution name - extract full name exactly as written, return empty string if not present")
    graduation_year: Optional[int] = Field(default=None, description="Graduation year - extract year of graduation or completion if mentioned")
    gpa: str = Field(default="", description="GPA if mentioned - extract GPA exactly as written (e.g., '3.8/4.0' or '3.8'), return empty string if not present")
    honors: str = Field(default="", description="Honors or distinctions - extract honors like 'Summa Cum Laude', 'Dean's List', 'With Distinction', etc., return empty string if not mentioned")
    
    @field_validator('graduation_year', mode='before')
    @classmethod
    def convert_empty_to_none_graduation_year(cls, v):
        """Convert empty strings to None for graduation_year field"""
        if v == '' or v is None:
            return None
        return v

class Certification(BaseModel):
    """Certification schema"""
    name: str = Field(default="", description="Certification name - extract full certification name exactly as written (e.g., 'CFA Level I', 'AWS Certified Solutions Architect'), return empty string if not present")
    issuer: str = Field(default="", description="Issuing organization - extract organization that issued the certification (e.g., 'CFA Institute', 'Amazon Web Services'), return empty string if not mentioned")
    year: Optional[int] = Field(default=None, description="Year obtained - extract year when certification was obtained if mentioned")
    level: str = Field(default="", description="Level if applicable - extract level (e.g., 'Level I', 'Level II', 'Associate', 'Professional'), return empty string if not applicable")
    
    @field_validator('year', mode='before')
    @classmethod
    def convert_empty_to_none_year(cls, v):
        """Convert empty strings to None for year field"""
        if v == '' or v is None:
            return None
        return v

# ============================================================================
# SEPARATE EXTRACTION SCHEMAS (Split for multi-step extraction)
# ============================================================================

class BasicInfoSchema(BaseModel):
    """Basic contact information schema"""
    name: str = Field(description="Full name of candidate")
    email: str = Field(default="", description="Email address - return empty string if not present")
    phone: str = Field(default="", description="Phone number - return empty string if not present")
    location: str = Field(default="", description="Geographic location/city - return empty string if not present")
    geography: Geography = Field(default=Geography.OTHER, description="Geographic market region")
    linkedin: str = Field(default="", description="LinkedIn profile URL - return empty string if not present")

class SummarySchema(BaseModel):
    """Professional summary schema"""
    summary: str = Field(default="", description="Professional summary or objective - return empty string if not present")
    experience_years: Optional[float] = Field(default=None, description="Total years of professional experience -  return 0 if not present")

class WorkExperienceSchema(BaseModel):
    """Work experience schema"""
    work_experience: List[WorkExperience] = Field(
        description="Complete work history with ALL positions including full-time, part-time, internships, contract roles, research positions - return empty list [] if not present",
        default_factory=list
    )

class EducationSchema(BaseModel):
    """Education schema"""
    education: List[Education] = Field(
        description="Complete education history",
        default_factory=list
    )

class SkillsSchema(BaseModel):
    """Skills and competencies schema"""
    skills: List[str] = Field(description="Technical and soft skills - extract ALL skills from entire resume including skills sections, work experience descriptions, and project descriptions - return empty list [] if not present", default_factory=list)
    programming_languages: List[str] = Field(description="Programming languages - extract ALL programming/scripting/query languages mentioned anywhere in resume (Python, Java, SQL, R, etc.) - return empty list [] if not present", default_factory=list)
    tools: List[str] = Field(description="Tools and software - extract ALL tools, platforms, software, frameworks, databases mentioned anywhere in resume (Excel, AWS, Docker, Git, Bloomberg Terminal, etc.) - return empty list [] if not present", default_factory=list)

class CertificationsSchema(BaseModel):
    """Certifications schema"""
    certifications: List[Certification] = Field(
        description="Professional certifications - extract ALL certifications, licenses, and credentials from entire resume including certifications sections and other mentions - return empty list [] if not present",
        default_factory=list
    )

class AdditionalInfoSchema(BaseModel):
    """Additional information schema"""
    sectors: List[str] = Field(
        description="Sectors/industries worked in - extract from work experience company names and descriptions, infer from context if needed - return empty list [] if not present",
        default_factory=list
    )
    languages: List[str] = Field(description="Languages spoken - extract ALL languages with proficiency levels if mentioned from entire resume - return empty list [] if not mentioned", default_factory=list)
    publications: List[str] = Field(description="Publications if any - extract ALL publications, papers, journal articles from entire resume - return empty list [] if not present", default_factory=list)
    awards: List[str] = Field(description="Awards and honors - extract ALL awards, honors, achievements, recognition from entire resume including education and work sections - return empty list [] if not present", default_factory=list)
    others: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured information not covered by other fields (volunteer work, memberships, patents, etc.) - return empty dict {} if not present"
    )

# ============================================================================
# COMBINED EXTRACTION SCHEMAS (For optimized multi-step extraction)
# ============================================================================

class BasicInfoSummarySchema(BaseModel):
    """Combined basic info and summary schema for optimized extraction"""
    # Basic Information
    name: str = Field(description="Full name of candidate")
    email: str = Field(default="", description="Email address - return empty string if not present")
    phone: str = Field(default="", description="Phone number - return empty string if not present")
    location: str = Field(default="", description="Geographic location/city - return empty string if not present")
    geography: Geography = Field(default=Geography.OTHER, description="Geographic market region")
    linkedin: str = Field(default="", description="LinkedIn profile URL - return empty string if not present")
    # Summary
    summary: str = Field(default="", description="Professional summary or objective - return empty string if not present")
    experience_years: Optional[float] = Field(default=None, description="Total years of professional experience - return 0 if not present")

class WorkEducationSchema(BaseModel):
    """Combined work experience and education schema for optimized extraction"""
    work_experience: List[WorkExperience] = Field(
        description="Complete work history with ALL positions including full-time, part-time, internships, contract roles, research positions - return empty list [] if not present",
        default_factory=list
    )
    education: List[Education] = Field(
        description="Complete education history - return empty list [] if not present",
        default_factory=list
    )

class SkillsCertificationsAdditionalSchema(BaseModel):
    """Combined skills, certifications, and additional info schema for optimized extraction"""
    skills: List[str] = Field(description="Technical and soft skills - extract ALL skills from entire resume including skills sections, work experience descriptions, and project descriptions - return empty list [] if not present", default_factory=list)
    programming_languages: List[str] = Field(description="Programming languages - extract ALL programming/scripting/query languages mentioned anywhere in resume (Python, Java, SQL, R, etc.) - return empty list [] if not present", default_factory=list)
    tools: List[str] = Field(description="Tools and software - extract ALL tools, platforms, software, frameworks, databases mentioned anywhere in resume (Excel, AWS, Docker, Git, Bloomberg Terminal, etc.) - return empty list [] if not present", default_factory=list)
    certifications: List[Certification] = Field(
        description="Professional certifications - extract ALL certifications, licenses, and credentials from entire resume including certifications sections and other mentions - return empty list [] if not present",
        default_factory=list
    )
    sectors: List[str] = Field(
        description="Sectors/industries worked in - extract from work experience company names and descriptions, infer from context if needed - return empty list [] if not present",
        default_factory=list
    )
    languages: List[str] = Field(description="Languages spoken - extract ALL languages with proficiency levels if mentioned from entire resume - return empty list [] if not mentioned", default_factory=list)
    publications: List[str] = Field(description="Publications if any - extract ALL publications, papers, journal articles from entire resume - return empty list [] if not present", default_factory=list)
    awards: List[str] = Field(description="Awards and honors - extract ALL awards, honors, achievements, recognition from entire resume including education and work sections - return empty list [] if not present", default_factory=list)
    others: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured information not covered by other fields (volunteer work, memberships, patents, etc.) - return empty dict {} if not present"
    )

class ResumeSchema(BaseModel):
    """
    Complete resume schema - combines all extracted features
    """
    # Basic Information
    name: str = Field(description="Full name of candidate")
    email: str = Field(default="", description="Email address - return empty string if not present")
    phone: str = Field(default="", description="Phone number - return empty string if not present")
    location: str = Field(default="", description="Geographic location/city - return empty string if not present")
    geography: Geography = Field(default=Geography.OTHER, description="Geographic market region")
    linkedin: str = Field(default="", description="LinkedIn profile URL - return empty string if not present")
    
    # Professional Summary
    summary: str = Field(default="", description="Professional summary or objective - return empty string if not present")
    experience_years: Optional[float] = Field(default=None, description="Total years of professional experience")
    
    # Work Experience (Detailed)
    work_experience: List[WorkExperience] = Field(
        description="Complete work history with ALL positions - return empty list [] if not present",
        default_factory=list
    )
    
    @field_validator('work_experience', mode='before')
    @classmethod
    def parse_work_experience_json(cls, v):
        """Parse JSON string to list for work_experience field"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return []
        return v
    
    # Education (Detailed)
    education: List[Education] = Field(
        description="Complete education history - return empty list [] if not present",
        default_factory=list
    )
    
    # Skills & Competencies
    skills: List[str] = Field(description="Technical and soft skills - extract from entire resume - return empty list [] if not present", default_factory=list)
    programming_languages: List[str] = Field(description="Programming languages - extract from entire resume - return empty list [] if not present", default_factory=list)
    tools: List[str] = Field(description="Tools and software - extract from entire resume - return empty list [] if not present", default_factory=list)
    
    # Certifications
    certifications: List[Certification] = Field(
        description="Professional certifications - extract from entire resume - return empty list [] if not present",
        default_factory=list
    )
    
    # Sector Information
    sectors: List[str] = Field(
        description="Sectors/industries worked in - extract from work experience - return empty list [] if not present",
        default_factory=list
    )
    
    # Languages
    languages: List[str] = Field(description="Languages spoken - extract from entire resume - return empty list [] if not present", default_factory=list)
    
    # Additional
    publications: List[str] = Field(description="Publications if any - extract from entire resume - return empty list [] if not present", default_factory=list)
    awards: List[str] = Field(description="Awards and honors - extract from entire resume - return empty list [] if not present", default_factory=list)
    
    # Others - Additional structured information
    others: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured information not covered by other fields - return empty dict if not present"
    )


# ============================================================================
# EXTRACTION STATE (For LangGraph workflow)
# ============================================================================

class ResumeExtractionState(TypedDict, total=False):
    """State for multi-step resume extraction workflow"""
    full_text: str
    extracted_data: Dict[str, Any]
    task_log: List[str]
    error: Optional[str]
    resume_uuid: Optional[str]
    file_path: Optional[str]


# ============================================================================
# UUID GENERATION SYSTEM (Reused from original)
# ============================================================================

class UUIDGenerator:
    """
    Generate custom UUIDs for resumes and chunks
    Formats:
    - resume-{hash}
    - chunk-{resume_uuid}-{section}-{index}
    """
    
    @staticmethod
    def generate_resume_uuid(file_name: str, timestamp: Optional[str] = None) -> str:
        """Generate resume UUID: resume-{hash(file_name + timestamp)}"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        hash_input = f"{file_name}_{timestamp}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        return f"resume-{hash_value}"
    
    @staticmethod
    def generate_chunk_uuid(resume_uuid: str, section: str, index: int) -> str:
        """Generate chunk UUID: chunk-{resume_uuid}-{section}-{index}"""
        return f"chunk-{resume_uuid}-{section}-{index}"


# ============================================================================
# TEXT SPLITTER MODULE
# ============================================================================

class TextSplitter:
    """
    LangChain-based text splitter for chunking documents
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize text splitter with configuration"""
        self.config = config or get_config()
        docloader_cfg = self.config.get_document_loader_config()
        
        # Initialize recursive character splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=docloader_cfg.chunk_size,
            chunk_overlap=docloader_cfg.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize markdown header splitter if enabled
        if docloader_cfg.use_markdown_splitter:
            self.markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            )
        else:
            self.markdown_splitter = None
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        docloader_cfg = self.config.get_document_loader_config()
        
        if docloader_cfg.text_splitter_type == "markdown" and self.markdown_splitter:
            # Try markdown splitting first
            all_chunks = []
            for doc in documents:
                try:
                    chunks = self.markdown_splitter.split_text(doc.page_content)
                    # Preserve metadata
                    for chunk in chunks:
                        chunk.metadata.update(doc.metadata)
                    all_chunks.extend(chunks)
                except Exception:
                    # Fallback to recursive splitting
                    chunks = self.recursive_splitter.split_documents([doc])
                    all_chunks.extend(chunks)
            return all_chunks
        else:
            # Use recursive character splitting
            return self.recursive_splitter.split_documents(documents)


# ============================================================================
# RESUME PARSER (LangChain-Based)
# ============================================================================

class ResumeParser:
    """
    LangChain-based resume parser using GPT-4o vision API
    Includes resume validation guardrail before processing
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize resume parser"""
        self.config = config or get_config()
        
        # Get configuration objects
        llm_cfg = self.config.get_llm_config()
        parser_cfg = self.config.get_parser_config()
        validation_cfg = self.config.get_validation_config()
        
        # Initialize vision document loader (GPT-4o vision-based)
        self.vision_loader = VisionDocumentLoader(config=self.config)
        self.text_splitter = TextSplitter(config=self.config)
        
        # Initialize document validator (guardrail)
        if validation_cfg.enable_resume_validation:
            self.validator = DocumentValidator(config=self.config)
        else:
            self.validator = None
        
        # Initialize LLM for extraction
        llm_kwargs = {
            "api_key": llm_cfg.api_key,
            "model": parser_cfg.llm_extraction_model,
            "temperature": llm_cfg.temperature
        }
        if llm_cfg.base_url:
            llm_kwargs["base_url"] = llm_cfg.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        self.structured_llm = self.llm.with_structured_output(ResumeSchema)
        self.uuid_generator = UUIDGenerator()
    
    # Combined extraction methods for optimized performance (reduces from 7 to 3 LLM calls)
    def _extract_basic_info_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic contact information and professional summary (combined for performance)"""
        return generic_resume_extraction(state, BasicInfoSummarySchema, 'extract_basic_info_summary', self.config)
    
    def _extract_work_education(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract work experience and education history (combined for performance)"""
        return generic_resume_extraction(state, WorkEducationSchema, 'extract_work_education', self.config)
    
    def _extract_skills_certifications_additional(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract skills, certifications, and additional information (combined for performance)"""
        return generic_resume_extraction(state, SkillsCertificationsAdditionalSchema, 'extract_skills_certifications_additional', self.config)
    
    def _merge_extracted_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge all extracted data and prepare for final schema creation"""
        return update_task_log_only(state, 'merge_extracted_data')
    
    def _build_extraction_graph(self) -> StateGraph:
        """Build LangGraph workflow for optimized multi-step resume extraction (3 combined steps instead of 7)"""
        graph_builder = StateGraph(ResumeExtractionState)
        
        # Add optimized combined extraction nodes (reduces from 7 to 3 LLM calls)
        graph_builder.add_node("extract_basic_info_summary", self._extract_basic_info_summary)
        graph_builder.add_node("extract_work_education", self._extract_work_education)
        graph_builder.add_node("extract_skills_certifications_additional", self._extract_skills_certifications_additional)
        graph_builder.add_node("merge_extracted_data", self._merge_extracted_data)
        
        # Define optimized sequential flow (3 steps instead of 7)
        graph_builder.add_edge(START, "extract_basic_info_summary")
        graph_builder.add_edge("extract_basic_info_summary", "extract_work_education")
        graph_builder.add_edge("extract_work_education", "extract_skills_certifications_additional")
        graph_builder.add_edge("extract_skills_certifications_additional", "merge_extracted_data")
        graph_builder.add_edge("merge_extracted_data", END)
        
        return graph_builder.compile()
    
    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Parse resume using GPT-4o vision API
        
        Process:
        Step 0: Validate document is a resume (guardrail)
        Step 1: Load document with vision loader
        Step 2: Split into chunks
        Step 3: Extract structured data with LLM (multi-step LangGraph workflow)
        Step 4: Generate UUIDs for chunks
        Step 5: Enrich with calculated fields
        
        Args:
            file_path: Path to resume file (PDF or DOCX)
            
        Returns:
            Dictionary containing:
            - resume_uuid: Unique identifier for resume
            - file_path: Original file path
            - file_name: Original file name
            - structured_data: Extracted structured data (ResumeSchema)
            - full_text: Combined text from all documents
            - chunks: List of document chunks with UUIDs
            - extraction_timestamp: When extraction occurred
            - validation_result: Document validation result (if validation enabled)
        """
        file_path = Path(file_path)
        
        # Step 0: Validate document is a resume (guardrail)
        validation_result = None
        if self.validator:
            validation_result = self.validator.validate_resume_preview(file_path)
            if not validation_result.is_resume:
                raise ValueError(
                    f"Document validation failed: {validation_result.reason}. "
                    f"Confidence: {validation_result.confidence:.2f}. "
                    f"Document type detected: {validation_result.document_type or 'unknown'}. "
                    f"Please upload a valid resume/CV document."
                )
        
        # Generate resume UUID
        resume_uuid = self.uuid_generator.generate_resume_uuid(file_path.name)
        
        # Step 1: Load document with GPT-4o vision
        documents = self.vision_loader.load_document(file_path)
        
        # Step 2: Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Combine full text
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Validate full_text extraction
        if not full_text or len(full_text.strip()) < 50:
            logger.warning(
                f"Text extraction may have failed. Full text length: {len(full_text)}, "
                f"Document pages: {len(documents)}"
            )
            if documents:
                first_page_len = len(documents[0].page_content) if documents[0].page_content else 0
                logger.debug(f"First page content length: {first_page_len}")
        
        # Step 3: Extract structured data using multi-step LangGraph workflow
        extraction_metadata = None
        try:
            # Initialize extraction state (TypedDict behaves like a dict)
            initial_state: ResumeExtractionState = {
                'full_text': full_text,
                'extracted_data': {},
                'task_log': [],
                'error': None,
                'resume_uuid': resume_uuid,
                'file_path': str(file_path)
            }
            
            # Build and execute extraction graph
            extraction_graph = self._build_extraction_graph()
            result_state = extraction_graph.invoke(initial_state)
            
            # Extract structured data from result state (LangGraph returns dict)
            extracted_data = result_state.get('extracted_data', {})
            
            # Provide default for name if missing (only required field)
            if 'name' not in extracted_data or not extracted_data.get('name'):
                extracted_data['name'] = 'Unknown'
                logger.warning("Name field was missing, using 'Unknown' as default")
            
            # Convert phone to string if it's an integer (LLM may return phone as integer)
            if 'phone' in extracted_data and isinstance(extracted_data['phone'], (int, float)):
                extracted_data['phone'] = str(int(extracted_data['phone']))
                logger.debug(f"Converted phone from {type(extracted_data['phone']).__name__} to string")
            
            # Create ResumeSchema from extracted data
            structured_data = ResumeSchema(**extracted_data)
            
            # Log extraction steps for debugging
            task_log = result_state.get('task_log', [])
            if task_log:
                logger.info(f"Extraction steps completed: {', '.join(task_log)}")
            
            # Log extraction statistics
            # logger.info("=" * 80)
            # logger.info("EXTRACTION SUMMARY")
            # logger.info("=" * 80)
            # logger.info(f"Work experiences extracted: {len(structured_data.work_experience)}")
            # logger.info(f"Education entries extracted: {len(structured_data.education)}")
            # logger.info(f"Skills extracted: {len(structured_data.skills)}")
            # logger.info(f"Programming languages extracted: {len(structured_data.programming_languages)}")
            # logger.info(f"Tools extracted: {len(structured_data.tools)}")
            # logger.info(f"Certifications extracted: {len(structured_data.certifications)}")
            # logger.info(f"Sectors extracted: {len(structured_data.sectors)}")
            # logger.info(f"Languages extracted: {len(structured_data.languages)}")
            # logger.info(f"Awards extracted: {len(structured_data.awards)}")
            # logger.info(f"Others keys extracted: {len(structured_data.others)}")
            # logger.info("=" * 80)
            
            # Validate critical fields
            
            # Log any errors but continue with partial extraction
            if result_state.get('error'):
                logger.warning(f"Extraction warnings: {result_state['error']}")
            
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Resume extraction failed: {e}") from e
        
        # Step 4: Generate UUIDs for chunks
        chunked_docs = []
        for idx, chunk in enumerate(chunks):
            # Determine section from metadata or content
            section = self._determine_section(chunk)
            
            chunk_uuid = self.uuid_generator.generate_chunk_uuid(
                resume_uuid=resume_uuid,
                section=section,
                index=idx
            )
            
            chunked_docs.append({
                'chunk_uuid': chunk_uuid,
                'resume_uuid': resume_uuid,
                'content': chunk.page_content,
                'metadata': chunk.metadata,
                'section': section
            })
        
        # Step 5: Enrich with calculated fields
        structured_data = self._enrich_data(structured_data)
        
        result = {
            'resume_uuid': resume_uuid,
            'file_path': str(file_path),
            'file_name': file_path.name,
            'structured_data': structured_data,
            'full_text': full_text,
            'chunks': chunked_docs,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Add extraction metadata if available
        if extraction_metadata is not None:
            result['extraction_metadata'] = extraction_metadata
        
        # Add validation result if available
        if validation_result:
            result['validation_result'] = {
                'is_resume': validation_result.is_resume,
                'confidence': validation_result.confidence,
                'reason': validation_result.reason,
                'document_type': validation_result.document_type
            }
        
        return result
    
    def _determine_section(self, chunk: Document) -> str:
        """Determine section name from chunk content or metadata"""
        # Check metadata first
        if 'Header 1' in chunk.metadata:
            return chunk.metadata['Header 1'].lower().replace(' ', '_')
        if 'Header 2' in chunk.metadata:
            return chunk.metadata['Header 2'].lower().replace(' ', '_')
        if 'Header 3' in chunk.metadata:
            return chunk.metadata['Header 3'].lower().replace(' ', '_')
        
        # Check content for common section headers
        content_lower = chunk.page_content.lower()[:200]
        if any(keyword in content_lower for keyword in ['experience', 'work', 'employment']):
            return 'experience'
        elif any(keyword in content_lower for keyword in ['education', 'degree', 'university']):
            return 'education'
        elif any(keyword in content_lower for keyword in ['skill', 'competenc', 'proficien']):
            return 'skills'
        elif any(keyword in content_lower for keyword in ['certification', 'certificate', 'cfa', 'frm']):
            return 'certifications'
        elif any(keyword in content_lower for keyword in ['summary', 'objective', 'profile']):
            return 'summary'
        else:
            return 'other'
    
    def _enrich_data(self, data: ResumeSchema) -> ResumeSchema:
        """Enrich extracted data with calculated fields"""
        # Calculate total experience from work experience dates if not already calculated
        if data.work_experience and (data.experience_years is None or data.experience_years == 0):
            total_months = 0
            for exp in data.work_experience:
                if exp.duration_months:
                    total_months += exp.duration_months
                elif exp.start_date and exp.end_date:
                    # Try to calculate duration from dates if duration_months not set
                    # This is a simple approximation - could be improved with date parsing
                    try:
                        # Extract years from date strings (simple heuristic)
                        start_year = None
                        end_year = None
                        for part in exp.start_date.split():
                            if part.isdigit() and len(part) == 4:
                                start_year = int(part)
                                break
                        if exp.end_date.lower() != 'present':
                            for part in exp.end_date.split():
                                if part.isdigit() and len(part) == 4:
                                    end_year = int(part)
                                    break
                        
                        if start_year and end_year:
                            months = (end_year - start_year) * 12
                            total_months += months
                    except Exception:
                        pass  # Skip if date parsing fails
            
            if total_months > 0:
                data.experience_years = round(total_months / 12, 1)
                logger.debug(f"Calculated experience_years: {data.experience_years} from work experience")
        
        # Extract sectors from work experience if not explicitly listed
        if not data.sectors and data.work_experience:
            sectors = set()
            for exp in data.work_experience:
                if exp.sector:
                    sectors.add(exp.sector)
            if sectors:
                data.sectors = list(sectors)
                logger.debug(f"Extracted sectors from work experience: {data.sectors}")
        
        # Ensure all list fields are properly initialized (not None)
        if data.skills is None:
            data.skills = []
        if data.programming_languages is None:
            data.programming_languages = []
        if data.tools is None:
            data.tools = []
        if data.sectors is None:
            data.sectors = []
        if data.languages is None:
            data.languages = []
        if data.publications is None:
            data.publications = []
        if data.awards is None:
            data.awards = []
        if data.others is None:
            data.others = {}
        
        return data
    
class ChromaDBStore:
    """
    Store resumes in ChromaDB with embeddings and UUIDs
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        
        # Get configuration objects
        chromadb_cfg = self.config.get_chromadb_config()
        llm_cfg = self.config.get_llm_config()
        
        self.db_path = Path(chromadb_cfg.db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Check write permissions
        if not os.access(self.db_path, os.W_OK):
            raise PermissionError(
                f"Database directory {self.db_path} is not writable. "
                f"Please check permissions. You may need to run: chmod -R u+w {self.db_path}"
            )
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at {self.db_path}: {e}. "
                f"Please ensure the directory exists and has write permissions."
            ) from e
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=chromadb_cfg.collection_name,
            metadata={"description": "Resume RAG database with UUID tracking"}
        )
        
        # Embedding model with config
        embedding_kwargs = {
            "api_key": llm_cfg.api_key,
            "model": llm_cfg.embedding_model
        }
        if llm_cfg.embedding_base_url:
            embedding_kwargs["base_url"] = llm_cfg.embedding_base_url
        
        self.embeddings = OpenAIEmbeddings(**embedding_kwargs)
        self.uuid_generator = UUIDGenerator()
        
        # Create LangChain Chroma wrapper (same as RAGSystem for compatibility)
        self.vectordb = Chroma(
            collection_name=chromadb_cfg.collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client
        )
    
    def store_resume(self, parsed_resume: Dict[str, Any]) -> str:
        """
        Store resume with chunks
        
        Args:
            parsed_resume: Dictionary from parse_resume() containing:
                - resume_uuid
                - structured_data
                - full_text
                - chunks (list of chunk dicts)
                
        Returns:
            resume_uuid string
        """
        resume_uuid = parsed_resume['resume_uuid']
        structured = parsed_resume['structured_data']
        chunks = parsed_resume.get('chunks', [])
        
        stored_ids = []
        
        # Store chunks
        for chunk in chunks:
            chunk_uuid = chunk.get('chunk_uuid')
            chunk_content = chunk.get('content', '')
            section = chunk.get('section', 'unknown')
            
            if not chunk_content.strip() or not chunk_uuid:
                continue
            
            # Generate embedding
            try:
                _ = self.embeddings.embed_query(chunk_content)
            except Exception as e:
                logger.error(f"Embedding generation failed for chunk {chunk_uuid}: {e}")
                continue
            
            # Metadata
            chunk_metadata = {
                'resume_uuid': resume_uuid,
                'chunk_uuid': chunk_uuid,
                'chunk_type': 'langchain',
                'section': section,
                'file_name': parsed_resume.get('file_name', ''),
                'name': structured.name if structured else 'Unknown',
                'geography': structured.geography.value if structured else 'Unknown',
                'experience_years': structured.experience_years if structured else 0.0,
                'sectors': ','.join(structured.sectors) if structured and structured.sectors else '',
                'skills': ','.join(structured.skills[:10]) if structured and structured.skills else '',
            }
            
            # Add original metadata
            original_metadata = chunk.get('metadata', {})
            if 'page' in original_metadata:
                chunk_metadata['page'] = original_metadata['page']
            if 'source' in original_metadata:
                chunk_metadata['source'] = original_metadata['source']
            
            try:
                # Use LangChain Chroma wrapper for storage (compatible with RAGSystem retrieval)
                doc = Document(page_content=chunk_content, metadata=chunk_metadata)
                self.vectordb.add_documents(
                    documents=[doc],
                    ids=[chunk_uuid]
                )
                stored_ids.append(chunk_uuid)
            except Exception as e:
                logger.error(f"Failed to store chunk {chunk_uuid}: {e}")
        
        # Also store section-based chunks for backward compatibility
        sections = {
            'header': self._format_header(structured),
            'summary': structured.summary or "",
            'experience': self._format_experience(structured.work_experience),
            'education': self._format_education(structured.education),
            'skills': self._format_skills(structured),
            'certifications': self._format_certifications(structured.certifications),
        }
        
        sections_stored = 0
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue
            
            chunk_index = len([c for c in stored_ids if c.startswith(f"chunk-{resume_uuid}-{section_name}")])
            chunk_uuid = self.uuid_generator.generate_chunk_uuid(resume_uuid, section_name, chunk_index)
            
            try:
                embedding = self.embeddings.embed_query(section_text)
            except Exception:
                continue
            
            metadata = {
                'resume_uuid': resume_uuid,
                'chunk_uuid': chunk_uuid,
                'section': section_name,
                'chunk_type': 'section',
                'name': structured.name,
                'geography': structured.geography.value,
                'experience_years': structured.experience_years,
                'sectors': ','.join(structured.sectors),
                'skills': ','.join(structured.skills[:10]),
            }
            
            try:
                # Use LangChain Chroma wrapper for storage
                doc = Document(page_content=section_text, metadata=metadata)
                self.vectordb.add_documents(
                    documents=[doc],
                    ids=[chunk_uuid]
                )
                stored_ids.append(chunk_uuid)
                sections_stored += 1
            except Exception as e:
                logger.error(f"Failed to store section chunk {chunk_uuid}: {e}")
        
        # CRITICAL: Always store full_text as fallback chunk if no sections were stored
        # This ensures data is searchable even if structured extraction failed
        full_text = parsed_resume.get('full_text', '')
        if full_text.strip() and (sections_stored == 0 or len(stored_ids) == 0):
            try:
                fallback_chunk_uuid = self.uuid_generator.generate_chunk_uuid(resume_uuid, 'full_text', 0)
                embedding = self.embeddings.embed_query(full_text)
                
                metadata = {
                    'resume_uuid': resume_uuid,
                    'chunk_uuid': fallback_chunk_uuid,
                    'section': 'full_text',
                    'chunk_type': 'fallback',
                    'file_name': parsed_resume.get('file_name', ''),
                    'name': structured.name if structured else 'Unknown',
                    'geography': structured.geography.value if structured else 'Unknown',
                    'experience_years': structured.experience_years if structured else 0.0,
                    'extraction_status': 'partial' if sections_stored == 0 else 'complete',
                }
                
                # Use LangChain Chroma wrapper for storage
                doc = Document(page_content=full_text, metadata=metadata)
                self.vectordb.add_documents(
                    documents=[doc],
                    ids=[fallback_chunk_uuid]
                )
                stored_ids.append(fallback_chunk_uuid)
                logger.info(
                    f"Stored fallback full_text chunk for {resume_uuid} "
                    f"(extraction_status: {metadata['extraction_status']})"
                )
            except Exception as e:
                logger.error(f"Failed to store fallback full_text chunk: {e}")
        
        return resume_uuid
    
    def _format_header(self, structured: ResumeSchema) -> str:
        """Format header section"""
        parts = [structured.name]
        if structured.email:
            parts.append(structured.email)
        if structured.phone:
            parts.append(structured.phone)
        if structured.location:
            parts.append(structured.location)
        return " | ".join(parts)
    
    def _format_experience(self, work_exp: List[WorkExperience]) -> str:
        """Format work experience section"""
        parts = []
        for exp in work_exp:
            exp_str = f"{exp.position} at {exp.company}"
            if exp.start_date:
                exp_str += f" ({exp.start_date} - {exp.end_date or 'Present'})"
            if exp.description:
                exp_str += f": {exp.description}"
            parts.append(exp_str)
        return "\n".join(parts)
    
    def _format_education(self, education: List[Education]) -> str:
        """Format education section"""
        parts = []
        for edu in education:
            edu_str = f"{edu.degree} in {edu.field} from {edu.institution}"
            if edu.graduation_year:
                edu_str += f" ({edu.graduation_year})"
            parts.append(edu_str)
        return "\n".join(parts)
    
    def _format_skills(self, structured: ResumeSchema) -> str:
        """Format skills section"""
        all_skills = []
        if structured.skills:
            all_skills.extend(structured.skills)
        if structured.programming_languages:
            all_skills.extend([f"Programming: {lang}" for lang in structured.programming_languages])
        if structured.tools:
            all_skills.extend([f"Tool: {tool}" for tool in structured.tools])
        return ", ".join(all_skills)
    
    def _format_certifications(self, certs: List[Certification]) -> str:
        """Format certifications section"""
        parts = []
        for cert in certs:
            cert_str = cert.name
            if cert.issuer:
                cert_str += f" ({cert.issuer})"
            if cert.year:
                cert_str += f" - {cert.year}"
            parts.append(cert_str)
        return ", ".join(parts)


def export_to_json(parsed_resumes: List[Dict], output_path: str = "parsed_resumes.json"):
    """Export parsed resumes to JSON"""
    export_data = []
    for resume in parsed_resumes:
        export_data.append({
            'resume_uuid': resume['resume_uuid'],
            'file_name': resume['file_name'],
            'file_path': resume['file_path'],
            'structured_data': resume['structured_data'].model_dump(),
            'extraction_timestamp': resume['extraction_timestamp'],
            'total_chunks': len(resume.get('chunks', []))
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)

def export_to_csv(parsed_resumes: List[Dict], output_path: str = "parsed_resumes.csv"):
    """Export parsed resumes to CSV summary"""
    rows = []
    for resume in parsed_resumes:
        structured = resume['structured_data']
        row = {
            'resume_uuid': resume['resume_uuid'],
            'file_name': resume['file_name'],
            'name': structured.name,
            'email': structured.email,
            'location': structured.location,
            'geography': structured.geography.value,
            'experience_years': structured.experience_years,
            'sectors': ', '.join(structured.sectors),
            'skills': ', '.join(structured.skills[:10]),
            'certifications': ', '.join([c.name for c in structured.certifications]),
            'education': '; '.join([f"{e.degree} {e.field}" for e in structured.education]),
            'others': json.dumps(structured.others) if structured.others else '',
            'extraction_timestamp': resume['extraction_timestamp']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8')
    return df


def main():
    """Main function to parse all resumes"""
    # Get configuration
    config = get_config()
    parser_cfg = config.get_parser_config()
    
    # Initialize components with config
    parser = ResumeParser(config=config)
    db_store = ChromaDBStore(config=config)
    
    # Process all resumes
    input_folder = Path(parser_cfg.input_folder)
    parsed_resumes = []
    
    for resume_file in sorted(input_folder.glob("*")):
        if resume_file.is_file() and resume_file.suffix.lower() in ['.pdf', '.docx', '.doc']:
            try:
                logger.info(f"Processing {resume_file.name}...")
                parsed = parser.parse_resume(resume_file)
                parsed_resumes.append(parsed)
                
                # Store in ChromaDB
                db_store.store_resume(parsed)
                logger.info(f"Successfully parsed and stored {resume_file.name}")
            except Exception as e:
                logger.error(f"Error processing {resume_file.name}: {e}", exc_info=True)
    
    # Export data
    export_to_json(parsed_resumes, output_path=parser_cfg.output_json)
    export_to_csv(parsed_resumes, output_path=parser_cfg.output_csv)
    
    logger.info(f"Processed {len(parsed_resumes)} resumes")
    logger.info(f"Exported to {parser_cfg.output_json} and {parser_cfg.output_csv}")
    
    return parsed_resumes

if __name__ == "__main__":
    main()
