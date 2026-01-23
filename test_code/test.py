"""
Test script for resume parser.
Demonstrates basic usage of the ResumeParser class.
"""

import logging
from pathlib import Path
from dotenv import load_dotenv

from resume_parser_langchain import ResumeParser
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)


def main():
    """Main function to test resume parsing."""
    # Initialize configuration
    config = get_config()
    logger.info("Configuration loaded successfully")
    
    # Initialize parser (sets up LangGraph workflow)
    parser = ResumeParser(config=config)
    logger.info("Parser initialized and ready")
    
    # Input: Change this to your document path
    doc_path = Path("/Users/siddharth.singh/Downloads/2025 DS Case Study/input/Chen Li (Alex).docx")
    logger.info(f"Processing document: {doc_path}")
    
    # Process: Run LangGraph workflow
    logger.info("Starting resume parsing workflow")
    result = parser.parse_resume(str(doc_path))
    logger.info("Resume parsing completed successfully")
    
    # Display basic results
    if result and 'structured_data' in result:
        structured = result['structured_data']
        logger.info(f"Extracted candidate: {structured.name}")
        logger.info(f"Experience: {structured.experience_years} years")
        logger.info(f"Location: {structured.location}")
    
    return result


if __name__ == "__main__":
    main()