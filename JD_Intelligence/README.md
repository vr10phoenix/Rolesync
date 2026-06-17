# Job Description Extractor
## Purpose:
An AI utility that is developed to extract the job description using Google Gemini 2.5 flash and LangChain that does deep analysis into the true essence and gives Json file in format of one of the example below : 
```JSON```
    {
  "mandatory_skills": [
    "Python",
    "Django",
    "4+ years of software engineering experience"
  ],
  "preferred_skills": [
    "Stripe API experience"
  ],
  "hidden_requirements": [
    "Problem-solving skills",
    "Autonomy and self-sufficiency",
    "Adaptability to rapidly changing requirements",
    "Ability to work effectively with incomplete documentation",
    "Understanding of financial or billing concepts"
  ],
  "seniority_level": "Senior",
  "team_type": "Fast-moving startup core functional team, cross-functional collaboration with finance",
  "communication_requirement": "Translating complex technical concepts to non-technical stakeholders (finance team), cross-functional collaboration",
  "leadership_requirements": "Mentorship and onboarding of junior developers, informal leadership in problem-solving and navigating ambiguity",
  "learning_ability_requirements": "Rapidly learn and adapt to new challenges and daily changes, thrive in ambiguity and a fast-paced environment, self-sufficient problem-solving without extensive documentation",
  "role_weights": {
    "technical": 0.3,
    "experience": 0.25,
    "leadership": 0.15,
    "learning": 0.2,
    "communication": 0.1
  }
}

this block is further communicated into local Vector database for searching

