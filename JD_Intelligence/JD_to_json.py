import os
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

#Gemini API Key

os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"

# additional metric to get better results in knowledge graph
class RoleWeights(BaseModel):
    technical: float = Field(
        description="Weight for hard technical skills and tool knowledge."
    )
    experience: float = Field(
        description="Weight for seniority, domain knowledge, and past scale."
    )
    leadership: float = Field(
        description="Weight for management, mentorship, and driving projects."
    )
    learning: float = Field(
        description="Weight for adaptability, fast-paced learning, and ambiguity."
    )
    communication: float = Field(
        description="Weight for cross-functional teamwork and stakeholder management."
    )

#filtering important components
class JobDescriptionIntelligence(BaseModel):
    mandatory_skills: List[str] = Field(
        description="Explicitly stated non-negotiable hard and soft skills."
    )
    preferred_skills: List[str] = Field(
        description="Skills listed as 'nice to have', 'bonus', or 'preferred'."
    )
    hidden_requirements: List[str] = Field(
        description="Implied requirements not explicitly stated (e.g., if the role requires deploying to AWS, 'cloud infrastructure experience' is implied)."
    )
    seniority_level: str = Field(
        description="The true seniority level (e.g., Junior, Mid-Level, Senior, Staff, Lead). Infer this from years of experience or responsibilities if not explicitly stated."
    )
    team_type: str = Field(
        description="The nature of the team (e.g., cross-functional, highly autonomous, globally distributed, tight-knit startup team)."
    )
    communication_requirement: str = Field(
        description="The type of communication expected (e.g., asynchronous, client-facing, translating technical concepts to non-technical stakeholders)."
    )
    leadership_requirements: str = Field(
        description="Expectations around mentorship, managing others, or leading projects without formal authority."
    )
    learning_ability_requirements: str = Field(
        description="Cues indicating the need to rapidly learn new technologies, adapt to ambiguity, or thrive in a fast-paced environment."
    )
    role_weights: RoleWeights = Field(
        description="Analyze the tone and requirements of the JD to distribute weight across these 5 categories. The values must be floats between 0.0 and 1.0, and they MUST mathematically sum up to exactly 1.0."
    )

#Initolaizing the LLM
def analyze_job_description(jd_text: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Pydantic model to output JSON
    structured_llm = llm.with_structured_output(JobDescriptionIntelligence)

    # system prompt
    system_prompt = """
    You are an elite, highly perceptive Technical Recruiter. Your job is to read a job description 
    and extract the true essence of the role. Do not just look for keywords—read between the lines.
    
    Analyze the provided job description and extract the data strictly into the provided JSON schema.
    
    CRITICAL INSTRUCTION FOR 'role weights': 
    Analyze the overall JD and determine what is most important for this specific role. 
    Assign a decimal weight to each of the 5 categories (technical, experience, leadership, learning, communication). 
    These 5 values MUST sum to exactly 1.0. For example, if it's a junior role, 'learning' might be 0.40 and 'leadership' might be 0.05.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here is the Job Description:\n\n{jd_text}")
    ])

    # extraction chain
    extraction_chain = prompt | structured_llm

    # main chain
    result = extraction_chain.invoke({"jd_text": jd_text})
    
    # output json
    return result.model_dump_json(indent=2)

# Test 
if __name__ == "__main__":
    sample_jd = """
    We are looking for a Software Engineer to join our core billing team. You must have at least 
    4 years of experience with Python and Django. Experience with Stripe API is a huge plus. 
    You will be working closely with the finance team to explain discrepancies, so you need to be 
    able to talk to non-engineers. We are a fast-moving series A startup, so things break and change 
    daily—you'll need to figure things out on your own without a lot of documentation. You'll also 
    be expected to help onboard the junior devs we hire next quarter.
    """
    
    print("Extracting important features about the job : \n")
    json_output = analyze_job_description(sample_jd)
    print(json_output)
