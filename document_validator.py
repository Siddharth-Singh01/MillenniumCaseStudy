#!/usr/bin/env python
# coding: utf-8

"""
Document Validation Guardrail Module

This module provides document validation functionality to ensure uploaded documents
are actually resumes/CVs before processing. Uses LLM-based classification for
accurate detection and prevents processing of invalid document types.

Author: Siddharth Singh
"""

import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from config import get_config, AppConfig

# Configure logging
logger = logging.getLogger(__name__)


class DocumentValidationResult(BaseModel):
    """Result of document validation"""
    is_resume: bool = Field(description="Whether document is a resume/CV")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reason: str = Field(description="Reason for validation decision")
    document_type: Optional[str] = Field(default=None, description="Type of document if not a resume (e.g., 'invoice', 'letter', 'certificate')")


class DocumentValidator:
    """
    Validates if a document is a resume before processing
    Uses LLM-based classification for accurate detection
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize document validator"""
        self.config = config or get_config()
        
        # Get configuration
        validation_cfg = self.config.get_validation_config()
        llm_cfg = self.config.get_llm_config()
        
        # Initialize validation LLM (use cheaper model for validation)
        llm_kwargs = {
            "api_key": llm_cfg.api_key,
            "model": validation_cfg.validation_model,
            "temperature": 0.0  # Deterministic validation
        }
        if llm_cfg.base_url:
            llm_kwargs["base_url"] = llm_cfg.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        self.validation_llm = self.llm.with_structured_output(DocumentValidationResult)
        self.confidence_threshold = validation_cfg.validation_confidence_threshold
    
    def validate_resume_preview(self, file_path: Path) -> DocumentValidationResult:
        """
        Validate if document is a resume by analyzing a preview
        
        Args:
            file_path: Path to document file
            
        Returns:
            DocumentValidationResult with validation decision
        """
        file_path = Path(file_path)
        
        # Quick file-level checks
        quick_check = self._quick_file_check(file_path)
        if not quick_check['passes']:
            return DocumentValidationResult(
                is_resume=False,
                confidence=0.0,
                reason=quick_check['reason'],
                document_type=quick_check.get('document_type')
            )
        
        # Extract preview text (first page or sample)
        preview_text = self._extract_preview_text(file_path)
        
        if not preview_text or len(preview_text.strip()) < 50:
            return DocumentValidationResult(
                is_resume=False,
                confidence=0.0,
                reason="Document appears to be empty or unreadable",
                document_type=None
            )
        
        # LLM-based validation
        return self._validate_with_llm(preview_text, file_path.name)
    
    def _quick_file_check(self, file_path: Path) -> dict:
        """
        Quick file-level validation checks
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with 'passes' boolean and 'reason' string
        """
        # Check file exists
        if not file_path.exists():
            return {
                'passes': False,
                'reason': f"File does not exist: {file_path}",
                'document_type': None
            }
        
        # Check file extension
        valid_extensions = ['.pdf', '.docx', '.doc']
        if file_path.suffix.lower() not in valid_extensions:
            return {
                'passes': False,
                'reason': f"Unsupported file type: {file_path.suffix}. Supported types: {', '.join(valid_extensions)}",
                'document_type': None
            }
        
        # Check file size (reject very large files)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:  # 50MB limit
            return {
                'passes': False,
                'reason': f"File too large: {file_size_mb:.1f}MB (maximum 50MB)",
                'document_type': None
            }
        
        # Check file name patterns (optional heuristic)
        file_name_lower = file_path.name.lower()
        resume_keywords = ['resume', 'cv', 'curriculum', 'vitae']
        if any(keyword in file_name_lower for keyword in resume_keywords):
            # File name suggests it's a resume, but still validate content
            return {'passes': True, 'reason': 'File name suggests resume'}
        
        return {'passes': True, 'reason': 'File checks passed'}
    
    def _extract_preview_text(self, file_path: Path, max_chars: int = 2000) -> str:
        """
        Extract preview text from document (first page or sample)
        
        Args:
            file_path: Path to document file
            max_chars: Maximum characters to extract
            
        Returns:
            Preview text string
        """
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._extract_pdf_preview(file_path, max_chars)
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                return self._extract_docx_preview(file_path, max_chars)
            else:
                return ""
        except Exception as e:
            logger.warning(f"Failed to extract preview text from {file_path}: {e}")
            return ""
    
    def _extract_pdf_preview(self, pdf_path: Path, max_chars: int) -> str:
        """Extract preview text from PDF (first page)"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            if len(doc) > 0:
                first_page = doc[0]
                text = first_page.get_text()
                doc.close()
                return text[:max_chars]
            doc.close()
            return ""
        except ImportError:
            return ""
        except Exception:
            return ""
    
    def _extract_docx_preview(self, docx_path: Path, max_chars: int) -> str:
        """Extract preview text from DOCX (first paragraphs)"""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(docx_path)
            text_parts = []
            for para in doc.paragraphs[:10]:  # First 10 paragraphs
                if para.text.strip():
                    text_parts.append(para.text)
            preview = "\n".join(text_parts)
            return preview[:max_chars]
        except ImportError:
            return ""
        except Exception:
            return ""
    
    def _validate_with_llm(self, preview_text: str, file_name: str) -> DocumentValidationResult:
        """
        Validate document using LLM
        
        Args:
            preview_text: Preview text from document
            file_name: Name of the file
            
        Returns:
            DocumentValidationResult
        """
        prompt = f"""Analyze this document and determine if it is a resume/CV.

File name: {file_name}

Document preview (first {len(preview_text)} characters):
{preview_text}

A resume/CV typically contains:
- Personal information: Full name, contact details (email, phone, address, LinkedIn)
- Professional summary or objective statement
- Work experience/employment history: Company names, job titles, dates, responsibilities
- Education background: Degrees, institutions, graduation years, GPA
- Skills and qualifications: Technical skills, languages, certifications
- Professional sections: Awards, publications, projects, etc.

A resume is NOT:
- An invoice, receipt, or financial document
- A letter or email
- A certificate or diploma alone
- A contract or legal document
- A report or research paper
- A presentation or slide deck

Determine if this document is a resume/CV. Provide:
1. is_resume: true if it's a resume, false otherwise
2. confidence: Your confidence level (0.0 to 1.0)
3. reason: Brief explanation of your decision
4. document_type: If not a resume, what type of document it appears to be (e.g., 'invoice', 'letter', 'certificate', 'report')"""
        
        try:
            result = self.validation_llm.invoke(prompt)
            
            # Handle both dict and DocumentValidationResult return types
            if isinstance(result, dict):
                # Convert dict to DocumentValidationResult
                try:
                    result = DocumentValidationResult(**result)
                except Exception as e:
                    # Fallback: create with defaults from dict
                    result = DocumentValidationResult(
                        is_resume=bool(result.get('is_resume', True)),
                        confidence=float(result.get('confidence', 0.5)),
                        reason=str(result.get('reason', f'Validation conversion error: {e}')),
                        document_type=result.get('document_type')
                    )
            elif not isinstance(result, DocumentValidationResult):
                # Unexpected type, create default
                logger.warning(f"Unexpected validation result type: {type(result)}")
                result = DocumentValidationResult(
                    is_resume=True,
                    confidence=0.5,
                    reason=f"Unexpected result type: {type(result)}",
                    document_type=None
                )
            
            # Apply confidence threshold
            if result.confidence < self.confidence_threshold:
                result.is_resume = False
                result.reason = f"Low confidence ({result.confidence:.2f}): {result.reason}"
            
            return result
        except Exception as e:
            # Fallback: assume it's a resume if validation fails (to avoid blocking valid resumes)
            logger.warning(f"LLM validation failed: {e}. Proceeding with document.")
            return DocumentValidationResult(
                is_resume=True,
                confidence=0.5,
                reason=f"Validation error, proceeding: {str(e)}",
                document_type=None
            )
