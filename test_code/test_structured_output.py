"""
Test script to debug structured output extraction.
Tests basic structured output functionality before fixing the main extraction.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Simple test schema
class BasicInfoSchema(BaseModel):
    name: str = Field(description="Full name")
    email: str = Field(default=None, description="Email address")
    location: str = Field(default=None, description="Location")

# Work Experience schema
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

class WorkExperienceSchema(BaseModel):
    """Work experience schema"""
    work_experience: List[WorkExperience] = Field(
        description="Complete work history with ALL positions including full-time, part-time, internships, contract roles, research positions - return empty list [] if not present",
        default_factory=list
    )

# Education schema
class Education(BaseModel):
    """Education schema"""
    degree: str = Field(default="", description="Degree - extract full degree name (Bachelor's, Master's, MBA, PhD, etc.) or abbreviations (BS, BA, MS, MA, MBA, PhD) as written, return empty string if not present")
    field: str = Field(default="", description="Field of study - extract major, specialization, or field (e.g., Computer Science, Business Administration, Engineering), return empty string if not present")
    institution: str = Field(default="", description="University/Institution name - extract full name exactly as written, return empty string if not present")
    graduation_year: Optional[int] = Field(default=None, description="Graduation year - extract year of graduation or completion if mentioned")
    gpa: str = Field(default="", description="GPA if mentioned - extract GPA exactly as written (e.g., '3.8/4.0' or '3.8'), return empty string if not present")
    honors: str = Field(default="", description="Honors or distinctions - extract honors like 'Summa Cum Laude', 'Dean's List', 'With Distinction', etc., return empty string if not mentioned")

class EducationSchema(BaseModel):
    """Education schema"""
    education: List[Education] = Field(
        description="Complete education history",
        default_factory=list
    )

# Skills schema
class SkillsSchema(BaseModel):
    """Skills and competencies schema"""
    skills: List[str] = Field(description="Technical and soft skills - extract ALL skills from entire resume including skills sections, work experience descriptions, and project descriptions - return empty list [] if not present", default_factory=list)
    programming_languages: List[str] = Field(description="Programming languages - extract ALL programming/scripting/query languages mentioned anywhere in resume (Python, Java, SQL, R, etc.) - return empty list [] if not present", default_factory=list)
    tools: List[str] = Field(description="Tools and software - extract ALL tools, platforms, software, frameworks, databases mentioned anywhere in resume (Excel, AWS, Docker, Git, Bloomberg Terminal, etc.) - return empty list [] if not present", default_factory=list)

# Additional info schema
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

# Initialize model
model = ChatOpenAI(
    model="aws/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    openai_api_key='e67aabc3-4d2c-47b7-83bd-b3fd4ccc40be',
    openai_api_base="https://public-api.grabgpt.managed.catwalk-k8s.stg-myteksi.com/unified/v1/"
)

# Sample resume text (from Chen Li resume)
resume_text = """
# Chen Li (Alex)

Mobile: +852 65478923  
Email: Alex_chen2024@gmail.com

# EDUCATION

## University of Hong Kong Hong Kong
Master of Science in Financial Technology Sept.2021 – Jul.2023  
Related Courses: Algorithmic Trading, Quantitative Risk Management, Machine Learning in Trading and Finance

## University of Hong Kong Hong Kong
Bachelor of Business Administration (Finance) Sept.2017 – Jul.2021  
Related Courses: Econometrics, Statistics

# WORKING EXPERIENCE

## Bank of China Hong Kong
Investment Analyst June.2021 – Sep.2023

Conduct comprehensive quantitative analysis on large-scale Asian equity datasets to provide statistical support for portfolio managers. Analyze market trends, sector rotations, and statistical anomalies, delivering actionable insights for investment decision-making and strategy optimization.

Independently developed, constructed and actively managed a multi-factor quantitative model focusing on Greater China technology equities, achieving superior risk-adjusted returns with Sharpe ratio of 1.8+ over 12-month period.

Established automated market surveillance system to monitor real-time market conditions, news sentiment, and macro events to identify investment opportunities and manage portfolio risk exposures.

Conducted rigorous backtesting and performance attribution analysis of systematic trading strategies, utilizing advanced statistical methods and machine learning techniques to enhance strategy performance and risk management.

## Investment Intern Dec.2019 – May.2021

Performed comprehensive research on Asian technology and healthcare sectors, covering 15+ sub-industries and 35+ companies across Greater China markets. Supported investment team by conducting expert interviews, analyzing financial statements, building DCF and relative valuation models, and preparing detailed investment research reports for portfolio review.

## Chinese University of Hong Kong Hong Kong
Research Assistant (Economics Area) Mar.2019 – Mar.2021

Efficiently processed and analyzed over 200,000 historical Asian market firm-level data points using Python and R, conducting comprehensive statistical analysis for academic research. Implemented advanced data processing techniques resulting in 25x improvement in research team computational efficiency.

Utilized Python and GIS mapping tools to analyze regional economic activity data from satellite imagery, generating 400+ analytical outputs including statistical reports and visualization packages for publication.

# Work Experience

## Meridian Asia Capital Jiangxi, China

**Finance Market Department Intern (Chinese Bond and Equity Market)** Jul.2019 – Aug.2019

• Conducted thorough investigation and analysis of equity information, resulting in the preparation of two comprehensive Equity Pledge Business Investigation Reports.

• Assisted in company research and played a key role in drafting Bill Credit Investigation Reports.

# Leadership and Involvement

## Fujia Aid Teaching Society Hong Kong

**Vice President & Head of Public Relations Team** Sep.2019 – Jun.2020

• Developed strategic initiatives for the organization and supervised two regional programs: Guangzhou Rural Education Project and Shenzhen Migrant Children Support Initiative.

• Led organizational marketing efforts across CUHK campus, increasing membership by 40% and creating educational video content to expand community engagement and awareness.

## University Swimming Club Hong Kong

**Head of Promotion Team** Sep.2018 – Sep.2019

• Managed promotional activities for the swimming club, collaborating with multiple student organizations to coordinate events including Inter-University Swimming Gala (80+ participants) and Water Safety Workshop series.

• Developed and maintained digital communication platforms including WeChat groups and Instagram accounts to enhance member engagement and information dissemination.

# Personal Information

**Languages:** English (fluent); Chinese (Mandarin: native, Cantonese: fluent)

**Computer Skills:** Python, R, SQL, Microsoft Office Suite (Excel with VBA, PowerPoint, Word), Bloomberg Terminal, WIND Database

**Awards:** Dean's List (4 semesters), Outstanding Academic Achievement Award, Top 5% in Provincial University Entrance Examination

**Personal Interests:** Swimming, Rock climbing, Violin performance, Documentary filmmaking (Regional Competition Winner)
"""

print("=" * 80)
print("TEST 1: Basic Info Extraction")
print("=" * 80)

structured_model_basic = model.with_structured_output(BasicInfoSchema)
prompt_basic = """Extract the following information from the text below. Extract ALL available fields.

Extract:
- Full name (REQUIRED)
- Email address (if present)
- Location (if present)

Text to extract from:
Name: John Doe, Email: john@example.com, Location: Singapore
"""

result_basic = structured_model_basic.invoke([HumanMessage(content=prompt_basic)])
print("Result dict:", result_basic)
print("Schema instance:", BasicInfoSchema(**result_basic))
print()

print("=" * 80)
print("TEST 2: Work Experience Extraction")
print("=" * 80)

structured_model_work = model.with_structured_output(WorkExperienceSchema)
prompt_work = f"""Extract ALL work experience entries from this resume.

IMPORTANT: Search the ENTIRE resume for work experience. Look in sections titled: "Experience", "Work Experience", "Employment", "Professional Experience", "Career History", "Working Experience", "Professional Background", or ANY section that contains job positions. Also check for internships, part-time roles, research positions, and volunteer work that may be listed separately.

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

CRITICAL: Return a list with ALL work positions found. Even if dates are unclear or some fields are missing, include the position with available information. Do not skip any positions. Be thorough and search the entire resume text.

Resume text:
{resume_text}
"""

result_work = structured_model_work.invoke([HumanMessage(content=prompt_work)])
print("Result dict:", result_work)
print("Number of work experiences extracted:", len(result_work.get('work_experience', [])))
if result_work.get('work_experience'):
    for i, exp in enumerate(result_work['work_experience'], 1):
        print(f"\nExperience {i}:")
        print(f"  Company: {exp.get('company', 'N/A')}")
        print(f"  Position: {exp.get('position', 'N/A')}")
        print(f"  Dates: {exp.get('start_date', 'N/A')} - {exp.get('end_date', 'N/A')}")
else:
    print("WARNING: No work experience extracted!")
print()

print("=" * 80)
print("TEST 3: Education Extraction")
print("=" * 80)

structured_model_edu = model.with_structured_output(EducationSchema)
prompt_edu = f"""Extract ALL education entries from this resume.

IMPORTANT: Look for education in sections titled: "Education", "Academic Background", "Qualifications", "Degrees", or similar. Extract EVERY degree, certification, or educational qualification mentioned, including undergraduate, graduate, and any additional degrees.

For EACH education entry found, extract:
- Degree (extract the full degree name: Bachelor's, Master's, MBA, PhD, Associate's, Diploma, Certificate, etc. Use abbreviations like BS, BA, MS, MA, MBA, PhD if that's what's written, if not present use empty string '')
- Field of study (extract the major, specialization, or field: Computer Science, Business Administration, Engineering, etc., if not present use empty string '')
- Institution name (extract the full university, college, or institution name, if not present use empty string '')
- Graduation year (extract the year of graduation or completion, if mentioned)
- GPA (extract GPA if mentioned (e.g., "3.8/4.0" or "3.8"), otherwise empty string '')
- Honors (extract any honors, distinctions, or awards: "Summa Cum Laude", "Dean's List", "With Distinction", etc., if mentioned, otherwise empty string '')

CRITICAL: Return a list with ALL education entries found. Include undergraduate, graduate, and any additional degrees or certifications. Even if some fields are missing, include the entry with available information.

Resume text:
{resume_text}
"""

result_edu = structured_model_edu.invoke([HumanMessage(content=prompt_edu)])
print("Result dict:", result_edu)
print("Number of education entries extracted:", len(result_edu.get('education', [])))
if result_edu.get('education'):
    for i, edu in enumerate(result_edu['education'], 1):
        print(f"\nEducation {i}:")
        print(f"  Degree: {edu.get('degree', 'N/A')}")
        print(f"  Field: {edu.get('field', 'N/A')}")
        print(f"  Institution: {edu.get('institution', 'N/A')}")
        print(f"  Year: {edu.get('graduation_year', 'N/A')}")
else:
    print("WARNING: No education extracted!")
print()

print("=" * 80)
print("TEST 4: Skills Extraction")
print("=" * 80)

structured_model_skills = model.with_structured_output(SkillsSchema)
prompt_skills = f"""Extract ALL skills from this resume.

IMPORTANT: Search the ENTIRE resume comprehensively. Look for skills in sections titled: "Skills", "Technical Skills", "Core Competencies", "Proficiencies", "Expertise", "Computer Skills", "Technical Expertise", "Tools & Technologies", or ANY section that lists skills. Also extract skills mentioned in work experience descriptions, project descriptions, education sections, and anywhere else in the resume.

Extract into three categories:

1. Technical and soft skills (list):
   - Include technical skills: data analysis, machine learning, project management, statistical analysis, quantitative analysis, etc.
   - Include soft skills: leadership, communication, teamwork, problem-solving, strategic thinking, etc.
   - Include domain-specific skills: financial modeling, risk management, software development, algorithmic trading, etc.
   - Extract ALL skills mentioned, even if they appear in different sections
   - Look for skills mentioned in bullet points, job descriptions, and project descriptions

2. Programming languages (list):
   - Extract ALL programming languages: Python, Java, C++, JavaScript, SQL, R, MATLAB, VBA, etc.
   - Include scripting languages (Python, Bash, PowerShell), query languages (SQL), and markup languages (HTML, XML) if relevant
   - Be thorough - extract from skills sections AND work experience descriptions AND project descriptions
   - Look for mentions like "Python", "R", "SQL", "Java", etc. in any context

3. Tools and software (list):
   - Extract ALL tools, platforms, and software mentioned anywhere in the resume
   - Examples: Excel, Tableau, AWS, Docker, Git, JIRA, Salesforce, Bloomberg Terminal, WIND Database, Microsoft Office Suite, etc.
   - Include frameworks, libraries, databases, cloud platforms, development tools, financial tools, etc.
   - Extract from both explicit skills lists AND work experience descriptions AND project descriptions
   - Look for tool names in job responsibilities, achievements, and technical sections
   - Include database systems (MySQL, PostgreSQL, MongoDB), cloud platforms (AWS, Azure, GCP), version control (Git, SVN), etc.

CRITICAL: Be comprehensive and thorough. Extract skills from:
- Dedicated skills sections (any format)
- Work experience job descriptions (read all bullet points carefully)
- Project descriptions
- Education sections (courses, coursework)
- Certifications sections
- Anywhere else skills, languages, or tools are mentioned

Return empty lists [] only if NO skills are found anywhere in the resume. When in doubt, include it.

Resume text:
{resume_text}
"""

result_skills = structured_model_skills.invoke([HumanMessage(content=prompt_skills)])
print("Result dict:", result_skills)
print(f"Skills extracted: {len(result_skills.get('skills', []))} items")
if result_skills.get('skills'):
    print("  Skills:", result_skills['skills'][:10])  # Show first 10
else:
    print("  WARNING: No skills extracted!")

print(f"\nProgramming languages extracted: {len(result_skills.get('programming_languages', []))} items")
if result_skills.get('programming_languages'):
    print("  Languages:", result_skills['programming_languages'])
else:
    print("  WARNING: No programming languages extracted!")

print(f"\nTools extracted: {len(result_skills.get('tools', []))} items")
if result_skills.get('tools'):
    print("  Tools:", result_skills['tools'][:10])  # Show first 10
else:
    print("  WARNING: No tools extracted!")
print()

print("=" * 80)
print("TEST 5: Additional Info Extraction (Languages, Awards, etc.)")
print("=" * 80)

structured_model_add = model.with_structured_output(AdditionalInfoSchema)
prompt_add = f"""Extract additional information from this resume.

IMPORTANT: Search the ENTIRE resume carefully and comprehensively for this information. Look in dedicated sections, work experience descriptions, education sections, and anywhere else information might be mentioned.

Extract:

1. Sectors/industries worked in (list):
   - Extract from work experience company names and descriptions
   - Common sectors: Finance, Technology, Healthcare, Consulting, Manufacturing, Retail, Education, Government, Non-profit, etc.
   - If not explicitly listed, infer from company names, job descriptions, and industry context
   - Look for sector mentions in company descriptions, job titles, or responsibilities
   - Return empty list [] only if no sector information can be determined from anywhere in the resume

2. Languages spoken (if mentioned, list):
   - Look for sections like "Languages", "Language Skills", "Language Proficiency", "Languages Spoken", or language mentions in other sections
   - Also check "Personal Information" sections or similar areas
   - Extract language names and proficiency levels if mentioned (e.g., "English (Native)", "Spanish (Fluent)", "Chinese (Mandarin: native, Cantonese: fluent)")
   - Include all languages mentioned with their proficiency levels
   - Return empty list [] if not mentioned anywhere

3. Publications (if any, list):
   - Look for sections like "Publications", "Research", "Papers", "Published Works", "Research Publications", or similar
   - Extract paper titles, journal names, conference names, or publication details
   - Include research papers, journal articles, conference papers, book chapters, etc.
   - Return empty list [] if not present

4. Awards and honors (if any, list):
   - Look for sections like "Awards", "Honors", "Achievements", "Recognition", "Accolades", "Distinctions", or similar
   - Extract award names, recognition titles, achievement descriptions, competition wins, etc.
   - Also check education section for academic honors (Dean's List, Summa Cum Laude, etc.)
   - Check work experience for professional awards and recognition
   - Include all types of awards: academic, professional, competition wins, scholarships, etc.
   - Return empty list [] if not present

5. Any other relevant information (dict):
   - Extract any other structured information not covered above
   - Examples: volunteer work, professional memberships, patents, licenses, hobbies/interests (if professional), leadership roles outside work, etc.
   - Look for sections like "Volunteer Work", "Professional Memberships", "Patents", "Interests", "Activities", etc.
   - Format as key-value pairs (e.g., {{"volunteer_work": "...", "memberships": "...", "interests": "..."}})
   - Return empty dict {{}} if not present

CRITICAL: Be thorough and comprehensive. Search the entire resume text from beginning to end. For list fields, return an empty list [] if no data is present. For dict fields, return an empty dict {{}} if no data is present. Do not return empty strings for list or dict fields. Extract all available information.

Resume text:
{resume_text}
"""

result_add = structured_model_add.invoke([HumanMessage(content=prompt_add)])
print("Result dict:", result_add)
print(f"Sectors extracted: {len(result_add.get('sectors', []))} items")
if result_add.get('sectors'):
    print("  Sectors:", result_add['sectors'])
else:
    print("  WARNING: No sectors extracted!")

print(f"\nLanguages extracted: {len(result_add.get('languages', []))} items")
if result_add.get('languages'):
    print("  Languages:", result_add['languages'])
else:
    print("  WARNING: No languages extracted!")

print(f"\nAwards extracted: {len(result_add.get('awards', []))} items")
if result_add.get('awards'):
    print("  Awards:", result_add['awards'])
else:
    print("  WARNING: No awards extracted!")

print(f"\nOthers extracted: {len(result_add.get('others', {}))} keys")
if result_add.get('others'):
    print("  Others:", result_add['others'])
else:
    print("  Others: {{}} (empty dict)")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
