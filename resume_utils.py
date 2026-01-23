"""
Utility functions for resume extraction.

This module provides generic extraction functions used by the LangGraph workflow
to extract structured data from resume text using LLM models.

Similar to authenticity_utils.py - handles all complex extraction logic.

Author: Siddharth Singh
"""

import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config import AppConfig
from langchain_openai import ChatOpenAI

# Configure logging
logger = logging.getLogger(__name__)


def _initialize_schema_fields_with_defaults(schema_class: type[BaseModel]) -> Dict[str, Any]:
    """
    Initialize all schema fields with appropriate default values.
    
    This ensures that even if extraction fails, all fields are present in the result.
    
    Args:
        schema_class: Pydantic schema class
        
    Returns:
        Dictionary with all schema fields initialized to their defaults
    """
    defaults = {}
    for field_name, field_info in schema_class.model_fields.items():
        # Check if field has a default_factory
        if field_info.default_factory is not None:
            defaults[field_name] = field_info.default_factory()
        # Check if field has a default value
        elif field_info.default is not None:
            defaults[field_name] = field_info.default
        else:
            # Infer default from type annotation
            annotation = field_info.annotation
            annotation_str = str(annotation)
            
            # Handle Optional types
            if 'Optional' in annotation_str or 'Union' in annotation_str:
                defaults[field_name] = None
            # Handle List types
            elif 'List' in annotation_str or 'typing.List' in annotation_str:
                defaults[field_name] = []
            # Handle Dict types
            elif 'Dict' in annotation_str or 'typing.Dict' in annotation_str:
                defaults[field_name] = {}
            # Handle string types
            elif 'str' in annotation_str:
                defaults[field_name] = ""
            # Default to None for other types
            else:
                defaults[field_name] = None
    
    return defaults


def load_llm(api_key: str, base_url: str = None, model_name: str = "gpt-4o", temperature: float = 0):
    """
    Initialize and return a ChatOpenAI LLM instance with specified configuration.
    
    Args:
        api_key: API key for the LLM service
        base_url: Optional base URL for custom API endpoints
        model_name: Name of the model to use
        temperature: Sampling temperature (0.0 for deterministic output)
        
    Returns:
        ChatOpenAI instance configured with the provided parameters
    """
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def generic_resume_extraction(
    state: Dict[str, Any],
    schema_class: type[BaseModel],
    function_name: str,
    config: AppConfig
) -> Dict[str, Any]:
    """
    Generic function to handle all resume extraction steps.
    
    This function performs structured extraction from resume text using an LLM
    with a specified Pydantic schema. It handles text truncation, LLM invocation,
    and data merging into the extraction state.
    
    Similar to generic_claims_analysis in authenticity_utils.py
    
    Args:
        state: Current state dictionary containing:
            - full_text: Complete resume text
            - extracted_data: Previously extracted data (dict)
            - task_log: List of completed extraction steps
        schema_class: Pydantic schema class defining the extraction structure
        function_name: Name of the extraction function (e.g., 'extract_basic_info')
        config: Application configuration object
        
    Returns:
        Updated state dictionary with:
            - task_log: Updated list including the current extraction step
            - extracted_data: Merged extracted data including new extraction results
    """
    node_id = function_name
    logger.debug(f"Starting extraction step: {node_id}")
    
    current_log = list(state.get('task_log', []))
    current_log.append(node_id)
    
    full_text = state.get('full_text', '')
    
    # Use full text without truncation - let LLM handle the entire resume
    text_to_use = full_text
    
    # Get parser config for LLM model name (no longer using max_length for truncation)
    parser_cfg = config.get_parser_config()
    
    # Get system prompt and extraction prompt for this extraction step
    system_prompt = _get_system_prompt_for_function(function_name)
    extraction_prompt = _get_prompt_for_function(function_name, text_to_use)
    
    # Get LLM configuration
    llm_cfg = config.get_llm_config()
    
    # Initialize LLM
    llm = load_llm(
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.base_url,
        model_name=parser_cfg.llm_extraction_model,
        temperature=llm_cfg.temperature
    )
    
    # Use structured output with json_mode for reliable extraction with custom API gateways
    structured_llm = llm.with_structured_output(schema_class, method="json_mode")
    
    try:
        # Invoke LLM with SystemMessage + HumanMessage pattern for better extraction
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=extraction_prompt)
        ]
        result = structured_llm.invoke(messages)
        
        logger.debug(f"LLM invocation completed for {node_id}. Result type: {type(result)}")
        
        # Handle both dict and Pydantic model return types
        # Use model_dump() for consistent dict access regardless of return type
        if isinstance(result, dict):
            extracted_data = result.copy()
        else:
            # Pydantic model - convert to dict
            extracted_data = result.model_dump()
        
        # Ensure all schema fields are present
        schema_fields = set(schema_class.model_fields.keys())
        result_fields = set(extracted_data.keys())
        
        # Fill missing string fields with empty string
        for field_name in schema_fields - result_fields:
            field_info = schema_class.model_fields[field_name]
            if hasattr(field_info, 'annotation'):
                annotation_str = str(field_info.annotation)
                if 'str' in annotation_str and 'Optional' not in annotation_str:
                    extracted_data[field_name] = ""
        
        # Transform empty strings to proper types for list/dict fields
        # This handles cases where LLM returns empty strings instead of empty lists/dicts
        for field_name, field_info in schema_class.model_fields.items():
            if field_name in extracted_data:
                annotation = field_info.annotation
                annotation_str = str(annotation)
                
                # Convert empty string to empty list for List fields
                if 'List' in annotation_str or 'typing.List' in annotation_str:
                    if extracted_data[field_name] == '':
                        extracted_data[field_name] = []
                
                # Convert empty string to empty dict for Dict fields
                elif 'Dict' in annotation_str or 'typing.Dict' in annotation_str:
                    if extracted_data[field_name] == '':
                        extracted_data[field_name] = {}
        
        # Validate extracted data and log statistics
        schema_defaults = _initialize_schema_fields_with_defaults(schema_class)
        
        # Log extraction statistics
        for field_name, field_info in schema_class.model_fields.items():
            if field_name in extracted_data:
                value = extracted_data[field_name]
                annotation_str = str(field_info.annotation)
                
                # Log list field counts
                if 'List' in annotation_str or 'typing.List' in annotation_str:
                    if isinstance(value, list):
                        count = len(value)
                        if count > 0:
                            logger.info(f"{node_id}: Extracted {count} items for field '{field_name}'")
                        else:
                            logger.warning(f"{node_id}: Field '{field_name}' is empty list - expected data may be missing")
                    else:
                        logger.warning(f"{node_id}: Field '{field_name}' is not a list (got {type(value)})")
                
                # Log dict field counts
                elif 'Dict' in annotation_str or 'typing.Dict' in annotation_str:
                    if isinstance(value, dict):
                        count = len(value)
                        if count > 0:
                            logger.info(f"{node_id}: Extracted {count} keys for field '{field_name}'")
                        else:
                            logger.debug(f"{node_id}: Field '{field_name}' is empty dict")
                    else:
                        logger.warning(f"{node_id}: Field '{field_name}' is not a dict (got {type(value)})")
                
                # Log string fields that were extracted
                elif 'str' in annotation_str and value and value != schema_defaults.get(field_name, ""):
                    logger.debug(f"{node_id}: Extracted '{field_name}': {str(value)[:50]}...")
        
        # Ensure all schema fields are present (fill missing ones with defaults)
        for field_name, default_value in schema_defaults.items():
            if field_name not in extracted_data:
                extracted_data[field_name] = default_value
                logger.debug(f"Field {field_name} not in extraction result, using default: {default_value}")
        
        # Ensure all schema fields are present (fill missing ones with defaults)
        schema_defaults = _initialize_schema_fields_with_defaults(schema_class)
        for field_name, default_value in schema_defaults.items():
            if field_name not in extracted_data:
                extracted_data[field_name] = default_value
                logger.debug(f"Field {field_name} not in extraction result, using default: {default_value}")
        
        # Merge with existing extracted data
        extracted = state.get('extracted_data', {}).copy()
        extracted.update(extracted_data)
        
        # Log which fields were successfully extracted (non-default values)
        extracted_fields = [k for k, v in extracted_data.items() if v != schema_defaults.get(k)]
        logger.info(f"Completed extraction step: {node_id}. Extracted {len(extracted_fields)} fields: {extracted_fields}")
        
        return {
            "task_log": current_log,
            "extracted_data": extracted
        }
    except Exception as e:
        logger.error(f"Error in extraction step {node_id}: {e}", exc_info=True)
        
        # Initialize all schema fields with defaults even on failure
        # This ensures partial extraction doesn't lose previously extracted data
        schema_defaults = _initialize_schema_fields_with_defaults(schema_class)
        extracted = state.get('extracted_data', {}).copy()
        
        # Only add defaults for fields that don't already exist in extracted_data
        for field_name, default_value in schema_defaults.items():
            if field_name not in extracted:
                extracted[field_name] = default_value
                logger.debug(f"Initializing missing field {field_name} with default: {default_value}")
        
        # Return state with error logged but continue processing
        return {
            "task_log": current_log,
            "extracted_data": extracted,
            "error": f"Extraction step {node_id} failed: {str(e)}"
        }


def _get_system_prompt_for_function(function_name: str) -> str:
    """
    Generate system prompt with explicit JSON key instructions for a specific extraction function.
    
    Args:
        function_name: Name of the extraction function
        
    Returns:
        System prompt string with explicit JSON key instructions
    """
    system_prompts = {
        # Combined extraction prompts for optimized performance
        'extract_basic_info_summary': """You are a resume parser. Extract basic contact information AND professional summary.
You MUST use the following JSON keys exactly as written (SNAKE_CASE):
- name (REQUIRED)
- email
- phone
- location
- geography (one of: US, Europe, Asia-Pacific, Other)
- linkedin
- summary
- experience_years (number or null)""",
        
        'extract_work_education': """You are a resume parser. Extract ALL work experience AND education history.
You MUST use the following JSON keys exactly as written (SNAKE_CASE):
- work_experience (array of objects with: company, position, start_date, end_date, duration_months, location, description, sector)
- education (array of objects with: degree, field, institution, graduation_year, gpa, honors)""",
        
        'extract_skills_certifications_additional': """You are a resume parser. Extract skills, certifications, and additional information.
You MUST use the following JSON keys exactly as written (SNAKE_CASE):
- skills (array of strings)
- programming_languages (array of strings)
- tools (array of strings)
- certifications (array of objects with: name, issuer, year, level)
- sectors (array of strings)
- languages (array of strings)
- publications (array of strings)
- awards (array of strings)
- others (object/dict with key-value pairs)"""
    }
    
    return system_prompts.get(function_name, "You are a resume parser. Extract information using the provided schema.")


def _get_prompt_for_function(function_name: str, text: str) -> str:
    """
    Generate extraction prompt for a specific extraction function.
    
    Args:
        function_name: Name of the extraction function
        text: Resume text to extract from
        
    Returns:
        Formatted prompt string for LLM extraction
    """
    
    # Common instruction for all prompts
    json_instruction = """
Return the output as a JSON object matching the schema."""
    
    empty_string_instruction = """
    
CRITICAL: All fields are required. If data for a field is not present in the resume text, return an empty string '' for that field. Do not skip any fields."""
    
    prompts = {
        # Combined extraction prompts for optimized performance
        'extract_basic_info_summary': f"""Extract basic contact information AND professional summary from this resume.
{json_instruction}

Extract BOTH sets of information:

1. Basic Contact Information:
- Full name (REQUIRED)
- Email address (if present, otherwise empty string '')
- Phone number (if present, otherwise empty string '')
- Location/city (if present, otherwise empty string '')
- Geography region (US, Europe, Asia-Pacific, Other)
- LinkedIn URL (if present, otherwise empty string '')

2. Professional Summary:
- Professional summary or objective (if present, otherwise empty string '')
- Total years of professional experience (calculate from work experience dates, return null if not calculable)
{empty_string_instruction}

Resume text:
{text}
""",
        'extract_work_education': f"""Extract ALL work experience AND education entries from this resume.
{json_instruction}

IMPORTANT: Extract BOTH work experience and education comprehensively.

1. Work Experience:
Search the ENTIRE resume for work experience. Look in sections titled: "Experience", "Work Experience", "Employment", "Professional Experience", "Career History", "Working Experience", "Professional Background", or ANY section that contains job positions. Also check for internships, part-time roles, research positions, and volunteer work.

Extract EVERY position mentioned, including:
- Full-time positions
- Part-time positions
- Internships (even if labeled as "Intern" or "Internship")
- Contract positions
- Research positions
- Volunteer work (if it's professional experience)
- Consulting roles
- Freelance work

For EACH position found, extract:
- Company name (extract the full company name exactly as written, if not present use empty string '')
- Job title/position (extract the exact job title including any prefixes like "Senior", "Junior", "Lead", etc., if not present use empty string '')
- Start date (extract start date in any format found - month/year, year only, or full date, if not present use empty string '')
- End date (extract end date or "Present" if currently employed, if not present use empty string '')
- Duration in months (calculate from start and end dates if possible, otherwise leave as null)
- Location (city/country where the job was located, if mentioned, otherwise empty string '')
- Description (extract ALL responsibilities, achievements, and key accomplishments - be thorough and comprehensive, include all bullet points and details, if not present use empty string '')
- Sector/industry (identify the industry sector like "Finance", "Technology", "Healthcare", "Consulting", etc., if not explicitly stated infer from company name or description, if not present use empty string '')

2. Education:
Look for education in sections titled: "Education", "Academic Background", "Qualifications", "Degrees", or similar. Extract EVERY degree, certification, or educational qualification mentioned, including undergraduate, graduate, and any additional degrees.

For EACH education entry found, extract:
- Degree (extract the full degree name: Bachelor's, Master's, MBA, PhD, Associate's, Diploma, Certificate, etc. Use abbreviations like BS, BA, MS, MA, MBA, PhD if that's what's written, if not present use empty string '')
- Field of study (extract the major, specialization, or field: Computer Science, Business Administration, Engineering, etc., if not present use empty string '')
- Institution name (extract the full university, college, or institution name, if not present use empty string '')
- Graduation year (extract the year of graduation or completion if mentioned. Return null (not empty string) if not mentioned)
- GPA (extract GPA if mentioned (e.g., "3.8/4.0" or "3.8"), otherwise empty string '')
- Honors (extract any honors, distinctions, or awards: "Summa Cum Laude", "Dean's List", "With Distinction", etc., if mentioned, otherwise empty string '')

CRITICAL: Return ALL work positions and ALL education entries found. Even if dates are unclear or some fields are missing, include the entry with available information. Do not skip any positions or education entries. Be thorough and search the entire resume text.

{empty_string_instruction}

Resume text:
{text}
""",
        'extract_skills_certifications_additional': f"""Extract skills, certifications, AND additional information from this resume.
{json_instruction}

IMPORTANT: Extract ALL three categories comprehensively.

1. Skills and Competencies:
Search the ENTIRE resume comprehensively. Look for skills in sections titled: "Skills", "Technical Skills", "Core Competencies", "Proficiencies", "Expertise", "Computer Skills", "Technical Expertise", "Tools & Technologies", or ANY section that lists skills. Also extract skills mentioned in work experience descriptions, project descriptions, education sections, and anywhere else in the resume.

Extract into three categories:
- Technical and soft skills (list): Include technical skills (data analysis, machine learning, project management, statistical analysis, quantitative analysis, etc.) and soft skills (leadership, communication, teamwork, problem-solving, strategic thinking, etc.)
- Programming languages (list): Extract ALL programming languages: Python, Java, C++, JavaScript, SQL, R, MATLAB, VBA, etc. Include scripting languages, query languages, and markup languages if relevant
- Tools and software (list): Extract ALL tools, platforms, and software mentioned anywhere: Excel, Tableau, AWS, Docker, Git, JIRA, Salesforce, Bloomberg Terminal, WIND Database, Microsoft Office Suite, etc.

2. Certifications:
Search the ENTIRE resume for certifications. Look in sections titled: "Certifications", "Certificates", "Professional Certifications", "Licenses", "Credentials", "Qualifications", or similar. Also check if certifications are mentioned in education sections, work experience descriptions, or other parts of the resume.

Extract ALL professional certifications, licenses, and credentials, including:
- Professional certifications (CFA, FRM, PMP, CPA, etc.)
- Technical certifications (AWS Certified, Google Cloud Certified, Microsoft Certified, etc.)
- Industry-specific certifications
- Licenses (if professional licenses are mentioned)

For EACH certification found, extract:
- Certification name (extract the full certification name exactly as written, e.g., "CFA Level I", "AWS Certified Solutions Architect", if not present use empty string '')
- Issuing organization (extract the organization that issued the certification, e.g., "CFA Institute", "Amazon Web Services", if mentioned, otherwise empty string '')
- Year obtained (extract the year when the certification was obtained if mentioned. Return null (not empty string) if not mentioned)
- Level (extract the level if applicable, e.g., "Level I", "Level II", "Associate", "Professional", etc., otherwise empty string '')

3. Additional Information:
Search the ENTIRE resume carefully and comprehensively for this information. Look in dedicated sections, work experience descriptions, education sections, and anywhere else information might be mentioned.

Extract:
- Sectors/industries worked in (list): Extract from work experience company names and descriptions. Common sectors: Finance, Technology, Healthcare, Consulting, Manufacturing, Retail, Education, Government, Non-profit, etc. If not explicitly listed, infer from company names, job descriptions, and industry context. Return empty list [] only if no sector information can be determined.
- Languages spoken (list): Look for sections like "Languages", "Language Skills", "Language Proficiency", "Languages Spoken", or language mentions in other sections. Extract language names and proficiency levels if mentioned. Return empty list [] if not mentioned anywhere.
- Publications (list): Look for sections like "Publications", "Research", "Papers", "Published Works", "Research Publications", or similar. Extract paper titles, journal names, conference names, or publication details. Return empty list [] if not present.
- Awards and honors (list): Look for sections like "Awards", "Honors", "Achievements", "Recognition", "Accolades", "Distinctions", or similar. Extract award names, recognition titles, achievement descriptions, competition wins, etc. Also check education section for academic honors and work experience for professional awards. Return empty list [] if not present.
- Any other relevant information (dict): Extract any other structured information not covered above (volunteer work, professional memberships, patents, licenses, hobbies/interests if professional, leadership roles outside work, etc.). Format as key-value pairs. Return empty dict {{}} if not present.

CRITICAL: Be comprehensive and thorough. Extract skills from dedicated skills sections, work experience descriptions, project descriptions, education sections, certifications sections, and anywhere else skills, languages, or tools are mentioned. Return empty lists [] only if NO data is found. When in doubt, include it.

{empty_string_instruction}

Resume text:
{text}
"""
    }
    
    return prompts.get(function_name, f"Extract information from this resume:\n\n{empty_string_instruction}\n\n{text}")


def update_task_log_only(state: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """
    Update task log without performing any extraction.
    
    This is a placeholder function used in the LangGraph workflow for nodes
    that don't perform extraction but need to update the task log.
    
    Args:
        state: Current state dictionary
        node_id: Name of the node/step to log
        
    Returns:
        Updated state with task_log containing the new node_id
    """
    current_log = list(state.get('task_log', []))
    current_log.append(node_id)
    return {"task_log": current_log}
