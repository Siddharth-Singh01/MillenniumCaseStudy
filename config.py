#!/usr/bin/env python
# coding: utf-8
"""
Configuration Management Module

This module provides centralized configuration management for the resume parser application.
It reads configuration from environment variables and provides typed configuration objects
for different components (LLM, document loader, vision parser, validation, ChromaDB, etc.).

Author: Siddharth Singh
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


class LLMConfig(BaseModel):
    """LLM API configuration"""
    api_key: str = Field(description="LLM API key")
    base_url: Optional[str] = Field(default=None, description="Base URL for LLM vision API")
    embedding_base_url: Optional[str] = Field(default=None, description="Base URL for embedding API")
    model: str = Field(default="aws/global.anthropic.claude-sonnet-4-5-20250929-v1:0", description="LLM model name")
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model name")
    temperature: float = Field(default=0.0, description="Temperature for LLM")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens")


class DocumentLoaderConfig(BaseModel):
    """LangChain Document Loader configuration"""
    chunk_size: int = Field(default=1000, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=200, description="Chunk overlap for text splitting")
    use_markdown_splitter: bool = Field(default=True, description="Use MarkdownHeaderTextSplitter if document has markdown structure")
    text_splitter_type: str = Field(default="recursive", description="Text splitter type: 'recursive' or 'markdown'")


class VisionParserConfig(BaseModel):
    """Vision Parser configuration"""
    vision_model: str = Field(default="aws/global.anthropic.claude-sonnet-4-5-20250929-v1:0", description="Model for document image parsing")
    vision_dpi: int = Field(default=200, description="Image resolution for document conversion")
    vision_detail: str = Field(default="high", description="Image detail level: 'high' or 'low'")
    enable_vision_parsing: bool = Field(default=True, description="Enable vision-based document parsing")


class ValidationConfig(BaseModel):
    """Document validation configuration"""
    enable_resume_validation: bool = Field(default=True, description="Enable resume validation guardrail")
    validation_model: str = Field(default="azure/gpt-4o-mini", description="LLM model for validation (cheaper model)")
    validation_confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold for resume validation (0-1)")


class ChromaDBConfig(BaseModel):
    """ChromaDB configuration"""
    db_path: str = Field(default="./chroma_db", description="Path to ChromaDB database")
    collection_name: str = Field(default="resumes", description="Collection name")
    persist_directory: Optional[str] = Field(default=None, description="Persist directory")


class ResumeParserConfig(BaseModel):
    """Resume parser configuration"""
    input_folder: str = Field(default="input", description="Input folder for resumes")
    output_json: str = Field(default="./parsed_resume/parsed_resumes.json", description="Output JSON file")
    output_csv: str = Field(default="./parsed_resume/parsed_resumes.csv", description="Output CSV file")
    max_text_length: int = Field(default=50000, description="Maximum text length for LLM processing")
    # LLM Extraction configuration
    llm_extraction_model: str = Field(default="aws/global.anthropic.claude-sonnet-4-5-20250929-v1:0", description="LLM model for structured extraction")


class StreamlitConfig(BaseModel):
    """Streamlit app configuration"""
    page_title: str = Field(default="Millennium Resume Search Platform", description="Page title")
    page_icon: str = Field(default="🔍", description="Page icon")
    layout: str = Field(default="wide", description="Layout mode")
    retriever_k: int = Field(default=5, description="Number of documents to retrieve")
    suggestion_top_n: int = Field(default=5, description="Top N candidates to suggest")


class AppConfig(BaseSettings):
    """
    Main application configuration class.
    
    Reads configuration from environment variables (via .env file) and provides
    typed configuration objects for all application components.
    
    Uses pydantic-settings for environment variable parsing with type validation.
    """
    
    # LLM Configuration
    llm_api_key: str = Field(default="", description="LLM API key")
    llm_base_url: Optional[str] = Field(default=None, description="Base URL for LLM vision API")
    # Direct mapping for .env file (base_url in .env maps here)
    base_url: Optional[str] = Field(default=None, description="Base URL from .env (maps to llm_base_url)")
    llm_embedding_base_url: Optional[str] = Field(default=None, description="Base URL for embedding API")
    llm_model: str = Field(default="aws/global.anthropic.claude-sonnet-4-5-20250929-v1:0", description="LLM model")
    llm_embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")
    llm_temperature: float = Field(default=0.0, description="LLM temperature")
    
    # Document Loader Configuration
    document_loader_chunk_size: int = Field(default=1000, description="Chunk size for text splitting")
    document_loader_chunk_overlap: int = Field(default=200, description="Chunk overlap for text splitting")
    document_loader_use_markdown_splitter: bool = Field(default=True, description="Use MarkdownHeaderTextSplitter if document has markdown structure")
    document_loader_text_splitter_type: str = Field(default="recursive", description="Text splitter type: 'recursive' or 'markdown'")
    
    # Vision Parser Configuration
    vision_parser_model: str = Field(default="aws/global.anthropic.claude-sonnet-4-5-20250929-v1:0", description="Model for document image parsing")
    vision_parser_dpi: int = Field(default=300, description="Image resolution for document conversion")
    vision_parser_detail: str = Field(default="high", description="Image detail level: 'high' or 'low'")
    vision_parser_enabled: bool = Field(default=True, description="Enable vision-based document parsing")
    
    # Validation Configuration
    validation_enabled: bool = Field(default=True, description="Enable resume validation guardrail")
    validation_model: str = Field(default="azure/gpt-4o-mini", description="LLM model for validation")
    validation_confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold for resume validation")
        
    # ChromaDB Configuration
    chromadb_path: str = Field(default="./chroma_db", description="ChromaDB path")
    chromadb_collection_name: str = Field(default="resumes", description="ChromaDB collection name")
    
    # Resume Parser Configuration
    parser_input_folder: str = Field(default="input", description="Parser input folder")
    parser_output_json: str = Field(default="parsed_resumes.json", description="Parser output JSON")
    parser_output_csv: str = Field(default="parsed_resumes.csv", description="Parser output CSV")
    parser_llm_extraction_model: str = Field(default="aws/global.anthropic.claude-sonnet-4-5-20250929-v1:0", description="LLM model for structured extraction")
    
    # Streamlit Configuration
    streamlit_retriever_k: int = Field(default=5, description="Streamlit retriever k")
    streamlit_suggestion_top_n: int = Field(default=5, description="Streamlit suggestion top N")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        env_prefix = "" 
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration"""
        base_url = self.base_url or os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL") or self.llm_base_url
        embedding_base_url = self.llm_embedding_base_url or os.getenv("EMBEDDING_BASE_URL")
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or self.llm_api_key
        
        return LLMConfig(
            api_key=api_key,
            base_url=base_url,
            embedding_base_url=embedding_base_url,
            model=self.llm_model,
            embedding_model=self.llm_embedding_model,
            temperature=self.llm_temperature
        )
    
    def get_document_loader_config(self) -> DocumentLoaderConfig:
        """Get Document Loader configuration"""
        return DocumentLoaderConfig(
            chunk_size=self.document_loader_chunk_size,
            chunk_overlap=self.document_loader_chunk_overlap,
            use_markdown_splitter=self.document_loader_use_markdown_splitter,
            text_splitter_type=self.document_loader_text_splitter_type
        )
    
    def get_vision_parser_config(self) -> VisionParserConfig:
        """Get Vision Parser configuration"""
        return VisionParserConfig(
            vision_model=self.vision_parser_model,
            vision_dpi=self.vision_parser_dpi,
            vision_detail=self.vision_parser_detail,
            enable_vision_parsing=self.vision_parser_enabled
        )
    
    def get_validation_config(self) -> ValidationConfig:
        """Get Validation configuration"""
        return ValidationConfig(
            enable_resume_validation=self.validation_enabled,
            validation_model=self.validation_model,
            validation_confidence_threshold=self.validation_confidence_threshold
        )
    
    def get_chromadb_config(self) -> ChromaDBConfig:
        """Get ChromaDB configuration"""
        return ChromaDBConfig(
            db_path=self.chromadb_path,
            collection_name=self.chromadb_collection_name,
            persist_directory=self.chromadb_path
        )
    
    def get_parser_config(self) -> ResumeParserConfig:
        """Get resume parser configuration"""
        return ResumeParserConfig(
            input_folder=self.parser_input_folder,
            output_json=self.parser_output_json,
            output_csv=self.parser_output_csv,
            llm_extraction_model=self.parser_llm_extraction_model
        )
    
    def get_streamlit_config(self) -> StreamlitConfig:
        """Get Streamlit configuration"""
        return StreamlitConfig(
            retriever_k=self.streamlit_retriever_k,
            suggestion_top_n=self.streamlit_suggestion_top_n
        )


# Global config instance (singleton pattern)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get or create global configuration instance (singleton pattern).
    
    Returns:
        AppConfig instance with loaded configuration from environment variables
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
