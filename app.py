import os
import asyncio
import base64
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import traceback
import logging
from io import BytesIO
import tempfile
import time
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pdfplumber
from pydantic import BaseModel
from dotenv import load_dotenv

# Try to import Groq for real AI features
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not available - using enhanced simulation mode")

# Load environment variables
load_dotenv()

# In-memory storage for sessions
sessions = {}

# --- NEW CONSTANTS FOR STRICTNESS AND FLOW CONTROL ---
# No minimum response check for report generation; the LLM is always called.
ACTION_NEXT_Q = "__NEXT_STRUCTURED_QUESTION__" 
# --- END NEW CONSTANTS ---

# Initialize FastAPI app
app = FastAPI(
    title="Conversational AI Interview System",
    description="AI asks questions AND listens to your answers in real conversation",
    version="9.3.1" # Version addressing JSON structure failure
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory=None, packages=None, html=False, check_dir=True), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConversationalInterview")

# Configuration
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Initialize Groq client (SINGLE CLIENT FOR ALL SERVICES)
groq_client = None
if GROQ_AVAILABLE and GROQ_API_KEY:
    if len(GROQ_API_KEY) > 10:
        try:
            # Attempt to initialize Groq client using the environment variable
            groq_client = Groq(api_key=GROQ_API_KEY) 
            logger.info("✅ Groq client initialized - Real AI conversation enabled")
            logger.info("Models: Chat (llama-3.1-8b-instant) | Report (llama-3.1-8b-instant)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq client (Invalid key or network error): {e}")
            groq_client = None
    else:
        logger.error("❌ Groq API Key found but too short/invalid. Real AI features disabled.")
else:
    logger.info("ℹ️ Running with enhanced simulation mode")
# Models
class JobDescription(BaseModel):
    text: str
    role: str

class InterviewRequest(BaseModel):
    session_id: str
    job_description: Optional[JobDescription] = None
    
class CategoryScore(BaseModel):
    category: str
    score: float
    comment: str

class InterviewReport(BaseModel):
    evaluation: List[CategoryScore]
    overallScore: float
    recommendation: str

# Helper functions
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    try:
        with BytesIO(pdf_bytes) as pdf_stream:
            with pdfplumber.open(pdf_stream) as pdf:
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"{page_text}\n\n"
                    except Exception:
                        continue
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def parse_resume_content(resume_text: str) -> Dict:
    """Improved parsing logic to identify name, experience, and core technical skills."""
    if not resume_text:
        return {"candidateName": "Candidate", "skills": [], "experience": [], "education": []}

    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]

    # --- 1. Attempt to extract Name (Improved Heuristics) ---
    candidate_name = ""
    for line in lines[:6]:
        # Check for non-header lines that look like names (Title Case, short, not an email/URL/phone)
        if line == line.title() and 1 < len(line.split()) < 4 and len(line) > 4:
            lower_line = line.lower()
            if not any(keyword in lower_line for keyword in ['skill', 'education', 'experience', 'contact', 'linkedin', '@', 'phone', 'portfolio', 'email']):
                candidate_name = line
                break
    
    if not candidate_name:
        candidate_name = "Candidate"
        
    # --- 2. Extract Skills and Experience (Focus on Project Details) ---
    skill_keywords = [
        'python', 'javascript', 'java', 'react', 'node', 'sql', 'html', 'css',
        'typescript', 'angular', 'vue', 'django', 'flask', 'mongodb', 'mysql',
        'postgresql', 'aws', 'docker', 'kubernetes', 'git', 'linux', 'api', 
        'oop', 'rest', 'microservices', 'agile', 'scrum', 'backend', 'frontend',
        'machine learning', 'classification', 'regression', 'clustering', 'data science',
        'tensorflow', 'pytorch', 'c++', 'golang', 'devops', 'ci/cd', 'architecture'
    ]

    found_skills = set()
    experience_lines = []

    technical_verbs = ['developed', 'implemented', 'designed', 'optimized', 'managed', 'deployed', 'architected', 'led', 'reduced', 'increased', 'built', 'resolved', 'streamlined']
    project_indicators = ['project:', 'key contributions:', 'results:', 'achievements:']
    
    for line in lines:
        lower_line = line.lower()

        # Find skills
        for skill in skill_keywords:
            if re.search(r'\b' + re.escape(skill.split()[0]) + r'\b', lower_line):
                found_skills.add(skill)

        # Find detailed experience lines (more rigorous capture)
        if (
            any(exp in lower_line for exp in ['developer', 'engineer', 'manager', 'analyst', 'architect', 'lead']) or
            any(verb in lower_line for verb in technical_verbs) or
            any(indicator in lower_line for indicator in project_indicators)
        ) and len(line) > 25: # Must be a substantial line
            experience_lines.append(line)
            
    # Clean up and prioritize the most relevant skills
    clean_skills = sorted(list(set([s.title() for s in found_skills])))
    
    return {
        "candidateName": candidate_name,
        "skills": clean_skills[:15],
        "experience": experience_lines[:8], # Now these are more likely to be descriptive project lines
        "education": []
    }
    

def generate_interview_questions(resume_text: str, job_description: str = None, resume_parsed: Dict = None) -> List[Dict]:
    """
    Generates a list of interview questions tailored for the new flow: 
    Journey -> Project Deep Dive + JD Alignment -> Tech Stack Drill -> HR/Closing.
    """
    questions = []
    resume_lower = resume_text.lower() if resume_text else ""
    candidate_name = resume_parsed.get('candidateName', 'Candidate')
    job_role_title = resume_parsed.get('jobDescription', {}).get('role', 'the technical role')
    
    # --- 1. Professional Journey and Main Project Intro ---
    questions.append({
        "id": 1,
        "category": "Professional Journey & Introduction", 
        "question": f"Welcome, **{candidate_name}**! I'm your AI interviewer. Let's begin by tracing your professional journey. Please provide a high-level overview of your career path, then focus on **one single, most impactful project** from your resume. Tell me about its goal, your specific role, and the business impact you delivered.",
        "follow_ups": [
            "What critical skill did you develop or refine during that project?",
            "What was the biggest non-technical roadblock you overcame?"
        ]
    })

    # --- 2. Project Deep Dive (Architecture, Technical Decisions, and JD Strewing) ---
    if job_description and len(job_description.strip()) > 50:
        # JD is present: Strewing happens here.
        jd_alignment_detail = f"Given the requirements for **{job_role_title}**, specifically how did the **architecture** of your main project align with the **scalability and performance needs** of that role?"
        category_name = "Project Architecture & JD Strewing"
    else:
        # No JD: Standard technical deep dive
        jd_alignment_detail = f"Could you provide a detailed explanation of the **system architecture** of that project? Focus on the data flow, the primary technical trade-offs you made, and how you ensured its **stability in production**."
        category_name = "Project Architecture & Technical Deep Dive"
    
    questions.append({
        "id": 2,
        "category": category_name,
        "question": f"Thank you, **{candidate_name}**. Now, let's drill down into the project you mentioned. {jd_alignment_detail}",
        "follow_ups": [
            "What specific monitoring tools did you implement, and what was your biggest technical failure on that system?",
            "Detail the exact database indexing or caching strategy you used to optimize performance."
        ]
    })
    
    # --- 3. Core Technical Skills Drill (Tech Stack Switch/Escalation Phase) ---
    questions.append({
        "id": 3,
        "category": "Core Technical Skills & Escalation",
        "question": f"Great. Now, let's pivot to core technical concepts. Looking at your resume and the requirements of **{job_role_title}**, what is the **one technical stack (e.g., Python, Java, AWS, Frontend)** you feel most proficient and passionate about? I will now conduct a deep technical drill in that area, escalating from a low to a hard difficulty question.",
        "follow_ups": [
            "If you want to **switch technologies** at any point, just say 'switch to [New Tech Stack]', and I will adjust the questions.",
            "If you want to move to the final behavioral questions, say 'next question'."
        ]
    })
    
    # --- 4. HR, Behavioral, and Closing (General and Closing) ---
    questions.append({
        "id": 4,
        "category": "Behavioral & Closing",
        "question": "We've covered a lot of technical ground. Let's finish with some high-level questions. How does your long-term career vision align with the goals of **{job_role_title}** here, and what is one area of your professional skill set that you are actively trying to improve?",
        "follow_ups": [
            "If you were hired, what would be your single most important deliverable in the first 90 days?",
            "Do you have any questions for me about the role, the team, or the company?"
        ]
    })
    
    # Re-align IDs just in case the length changed
    for i, q in enumerate(questions):
        q['id'] = i + 1

    return questions

async def transcribe_audio_with_groq(audio_data: bytes) -> str:
    """Transcribe audio using Groq Whisper"""
    # NOTE: Uses the single initialized groq_client
    if not audio_data or len(audio_data) < 1000:
        return ""

    if not groq_client:
        # Enhanced simulation responses - randomized to prevent predictability
        responses = [
            "My experience is mainly in Python for machine learning. I'm exploring deep learning frameworks now.",
            "The most complex part was optimizing the database queries to handle large transactional loads efficiently.",
            "I would choose Java for its robustness in enterprise applications, especially for multi-threading performance.",
            "I'm looking to improve my knowledge of cloud infrastructure like AWS Lambda and Terraform.",
            "I enjoy the challenging part of converting a vague business requirement into a tangible technical design."
        ]
        import random
        return random.choice(responses)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            temp_filename = temp_file.name

        try:
            with open(temp_filename, "rb") as audio_file:
                # Transcription Model: whisper-large-v3
                transcription = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3", 
                    response_format="text",
                    language="en"
                )

            result = transcription.strip() if transcription else ""
            if len(result) > 5:
                logger.info(f"Real transcription: {result[:100]}...")
                return result
            else:
                return ""

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""

async def generate_conversational_response(user_text: str, context: Dict) -> Dict[str, str]:
    """
    Generate conversational AI response or return a signal to advance the structured question.
    NOTE: Uses the single initialized groq_client
    Returns: {"response": str, "action": str (text or ACTION_NEXT_Q)}
    """
    if not user_text or len(user_text.strip()) < 5:
        return {"response": "I'm sorry, I didn't catch that. Could you please repeat your response?", "action": "CONTINUE"}

    current_question = context.get('current_question', {})
    question_category = current_question.get('category', 'General')
    conversation_history = context.get('conversation', [])
    session_data = context.get('session_data', {})
    resume_parsed = session_data.get('resumeParsed', {})
    candidate_name = resume_parsed.get('candidateName', 'Candidate')
    
    # --- Check for explicit flow control requests (Simulation & Real) ---
    user_lower = user_text.lower()
    
    # --- 1. REPEAT/CLARIFICATION REQUEST (Highest Priority) ---
    repeat_phrases = ["repeat the question", "say that again", "i don't hear", "could you repeat", "didn't catch that", "can you repeat"]
    if any(phrase in user_lower for phrase in repeat_phrases):
        last_ai_message = next((item['content'] for item in reversed(conversation_history) if item['role'] == 'ai'), "Could you please clarify what you'd like me to repeat?")
        return {"response": last_ai_message, "action": "CONTINUE"}
        
    # --- 2. END INTERVIEW REQUEST (Critical) ---
    if any(phrase in user_lower for phrase in ["let's conclude", "i'm done", "i'm going", "end interview", "i can leave the interview", "i have some work", "sorry i have work"]):
        return {"response": f"I understand. Thank you for your time, {candidate_name}. That concludes our interview and we will generate your report now.", "action": ACTION_NEXT_Q}

    # --- 3. NEXT STRUCTURED QUESTION REQUEST (Critical) ---
    if any(phrase in user_lower for phrase in [
        "can't answer further", "proceed for further questions", "move on", "next question", 
        "skip this", "i don't know much about that", "i don't know", "next topic", 
        "ask me another question", "carry on", "change the topic", "let's talk about"
    ]) and current_question.get('id', 0) != 4: # Allow transition out of Q1-Q3
        return {"response": f"I understand, {candidate_name}. Let's conclude this topic and move to the next structured question.", "action": ACTION_NEXT_Q}
    
    # --- 4. TECH STACK SWITCH REQUEST (Specific to Q3) ---
    current_q_id = current_question.get('id')
    
    # We use a regex to look for pivot commands only during the Core Technical Skills drill (Q3)
    if current_q_id == 3:
        switch_match = re.search(r'(switch|pivot|change).*?(to|technologies|topic).*(?P<stack>\b\w+\b)', user_lower)
        if switch_match:
            new_stack = switch_match.group('stack').title()
            # Respond conversationally to acknowledge the switch, then let the LLM generate the first question in the new stack.
            return {"response": f"Understood, **{candidate_name}**. We are now pivoting our deep-dive focus to **{new_stack}**. Let's start with a foundational question in that domain.", "action": "CONTINUE"}


    # Third Priority: Simulation Mode Pivot Detection (If Groq is unavailable)
    if not groq_client:
        if "java coding questions" in user_lower or ("java" in user_lower and "coding" in user_lower):
            return {"response": f"Absolutely, {candidate_name}. Let's pivot to Java coding. Describe how you would implement a thread-safe singleton pattern in Java 8 or later.", "action": "CONTINUE"}
        
        # Simulation responses adapted to be friendly and drill down
        category_responses = {
            'Professional Journey & Introduction': [
                f"That's a very solid achievement, {candidate_name}! To strew that achievement: what was the specific complexity or data challenge that surprised you the most?",
                f"Thank you for that introduction, {candidate_name}. It sounds like you're aiming for system ownership. How does that project specifically prepare you for senior responsibilities?"
            ],
            'Project Architecture & JD Strewing': [
                f"That sounds like a demanding recovery, {candidate_name}! Let's drill down: what was the exact commit or configuration change that caused the failure, and what metrics did you use to get the 'all clear' after the fix?",
                f"Impressive work, {candidate_name}! Can you explain the *exact* data structure you chose to handle the memory constraints when processing large, real-time datasets?"
            ],
            'Project Architecture & Technical Deep Dive': [
                f"That sounds like a demanding recovery, {candidate_name}! Let's drill down: what was the exact commit or configuration change that caused the failure, and what metrics did you use to get the 'all clear' after the fix?",
                f"Impressive work, {candidate_name}! Can you explain the *exact* data structure you chose to handle the memory constraints when processing large, real-time datasets?"
            ],
            'Core Technical Skills & Escalation': [
                f"That's a solid explanation, {candidate_name}. Now, let's take that a layer deeper: why is it generally recommended to normalize the features before training a linear regression model, and what happens if you skip that step?",
                f"Very insightful, {candidate_name}. Based on your experience, how does **Python's GIL** affect high-performance data processing, and what *specific* library or approach do you use to work around it?"
            ],
            'Behavioral & Closing': [
                f"Those are excellent questions, {candidate_name}. To follow up on your point about team culture: how do you foster technical curiosity and continuous learning within a team?",
                f"That's a strong vision for your 90 days, {candidate_name}! Can you give me a specific technical goal and a non-technical goal you'd set for yourself in that period?"
            ],
            'General': [
                f"That's very insightful, {candidate_name}! Can you tell me more about how that experience shaped your approach?",
                f"Interesting perspective, {candidate_name}! What did you learn from that situation?"
            ]
        }
        
        import random
        responses = category_responses.get(question_category, category_responses['General'])
        return {"response": random.choice(responses), "action": "CONTINUE"}
    # --- End Simulation Mode ---

    # --- Real Groq-powered conversational response generation ---
    try:
        recent_context = ""
        if conversation_history:
            # Get the last 6 exchanges (3 pairs) for better context
            recent_exchanges = conversation_history[-6:] 
            for entry in recent_exchanges:
                role = "Interviewer" if entry.get("role") == "ai" else "Candidate"
                content = entry.get('content', '').replace('\n', ' ') 
                recent_context += f"{role}: {content}\n"
        
        resume_context = json.dumps(resume_parsed.get('skills', []))
        job_role = session_data.get('jobDescription', {}).get('role', 'a technical role')
        
        # Determine the technical stack focus based on the last few turns
        # This is a highly specialized prompt for the LLM to manage the difficulty and stack focus dynamically
        technical_drill_instruction = ""
        if current_q_id == 3:
            # CRITICAL INSTRUCTION for the drilling phase
            technical_drill_instruction = """
            ***TECHNICAL DRILL INSTRUCTIONS (Q3 ONLY)***
            1. **Determine Focus:** If the user's response mentions a technology (e.g., 'Python', 'AWS', 'Java'), immediately focus the conversation on that technology. If they ask to switch, acknowledge the switch and start a new drill in the new stack.
            2. **Difficulty Escalation:** Ensure your follow-up question is **more technically demanding** than the last one (Low -> Medium -> Hard).
            3. **Rigorous Check:** Your goal is to find the **absolute limit** of the candidate's knowledge by asking specific, rigorous questions about implementation details, performance trade-offs, or advanced concepts.
            """
        elif current_q_id == 4:
            # HR/Behavioral instruction
            technical_drill_instruction = "Focus your follow-up only on the HR, behavioral, or high-level strategic aspects of their answer. Avoid diving back into low-level code details."


        # --- CRITICAL SYSTEM PROMPT REVISION FOR TECHNICAL DRILL DOWN ---
        system_prompt = f"""You are a highly skilled, **friendly, rigorously professional, and demanding** AI interviewer specializing in **{job_role}**. Your primary goal is to conduct a **deep-drill, strewing conversation** to rigorously assess the candidate's core technical competence.

Candidate: **{candidate_name}**
Current Interview Phase: **{question_category}** (Structured Question ID: {current_q_id}/4)
Candidate's Key Skills from Resume (to guide initial drilling): {resume_context}

{technical_drill_instruction}

Recent conversation history:
{recent_context}

The candidate just responded: "{user_text}"

Your goal is to generate ONLY the next conversational line of questioning that is challenging and highly specific to their answer.

1.  **Tone:** Start with a friendly acknowledgment that **must include the candidate's name, {candidate_name}**.
2.  **Logic:** Ask a challenging, highly specific, technical follow-up that seeks to find the limits of their knowledge **based directly on their response ("{user_text}")**. Focus on **implementation specifics, edge cases, system trade-offs, or quantifiable metrics.**
3.  **Length Constraint:** Keep your entire response to **one or two concise, encouraging sentences (under 200 characters total)**.
"""
        # --- END SYSTEM PROMPT REVISION ---

        # Conversational Model: llama-3.1-8b-instant
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.85, 
            max_tokens=250 
        )

        ai_response = response.choices[0].message.content.strip()
        logger.info(f"Generated conversational response: {ai_response[:80]}...")
        
        # We assume if the user didn't ask to move on, the AI will continue the drill.
        return {"response": ai_response, "action": "CONTINUE"}

    except Exception as e:
        logger.error(f"AI response generation failed: {e}")
        return {"response": "That's really insightful! I'd love to follow up on that point to understand your thinking better.", "action": "CONTINUE"}

async def generate_interview_report(session_data: Dict) -> Dict:
    """
    Generates a structured final report using LLM based on conversation history.
    NOTE: Uses the single initialized groq_client
    """
    
    # --- GROQ CLIENT OFFLINE / INITIALIZATION FAILURE ---
    if not groq_client:
        logger.error("GROQ CLIENT UNAVAILABLE. Cannot generate dynamic report. Returning AI Offline error report.")
        # FALLBACK: Explicit error state when the client could not be initialized (e.g., missing API key)
        return {
            "evaluation": [
                {"category": "Communication", "score": 0.0, "comment": "AI client offline; report generation failed."},
                {"category": "Technical Depth", "score": 0.0, "comment": "AI client offline; report generation failed."},
                {"category": "Confidence", "score": 0.0, "comment": "AI client offline; report generation failed."},
                {"category": "Emotional Control", "score": 0.0, "comment": "AI client offline; report generation failed."},
                {"category": "Job Role Alignment", "score": 0.0, "comment": "AI client offline; report generation failed."}
            ],
            "overallScore": 0.0, 
            "recommendation": "AI Report Generation Failed: Groq client is not initialized or API Key is missing."
        }

    # Prepare data for LLM
    candidate_name = session_data['resumeParsed'].get('candidateName', 'Candidate')
    job_role = session_data['jobDescription'].get('role', 'Technical Role')
    
    transcript = "\n".join([f"{item['role'].title()}: {item['content']}" for item in session_data['conversation']])

    # --- CRITICAL: ENHANCED SYSTEM PROMPT FOR NON-HARDCODED, EVIDENCE-BASED ANALYSIS ---
    # NOTE: Changing the target job role to Senior/Lead for rigor, as requested by previous user queries
    system_prompt = f"""You are a senior technical hiring manager. Your task is to analyze the complete interview transcript and generate a final evaluation report. You are grading this candidate for a **Senior/Lead level technical role**, demanding **uncompromising standards**. Be extremely rigorous. A score of 7 or lower suggests serious performance gaps. A score of 5 or lower means a strong rejection.
    
    Interview Context:
    - Candidate Name: {candidate_name}
    - Job Role: {job_role}
    - Candidate Skills: {session_data['resumeParsed'].get('skills', [])}
    
    Interview Transcript (Use this as your sole evidence to justify scores):
    ---
    {transcript}
    ---
    
    Instructions:
    1. Score each category (Communication, Technical Depth, Confidence, Emotional Control, Job Role Alignment) from 1 to 10 (10 is best).
    2. THE JSON OUTPUT MUST CONTAIN ONE TOP-LEVEL KEY CALLED "evaluation" WHICH IS A LIST OF 5 CATEGORY OBJECTS.
    3. The 'Overall Score' must be the average of the five category scores, rounded to one decimal point.
    4. The 'Recommendation' must be a single sentence: 'Recommended for next round (Score X.X)' or 'Not recommended at this time (Score X.X)'.
    5. Provide an insightful 'Comment' for each category (under 15 words). **Crucially, ensure each comment is UNIQUE and SPECIFICALLY references an observation or quote from the transcript.**
    6. **OUTPUT ONLY THE RAW JSON OBJECT. DO NOT INCLUDE MARKDOWN FENCES (```json) OR ANY OTHER TEXT.**
"""
    # --- END ENHANCED SYSTEM PROMPT ---

    try:
        # Step 1: Call Groq for structured JSON response (using llama-3.1-8b-instant for reliability)
        # Report Model: llama-3.1-8b-instant
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.2, 
        )
        
        json_string = response.choices[0].message.content.strip()

        # Step 2: Parse the response with enhanced cleaning
        report = {}
        try:
            if json_string.startswith('```'):
                json_string = json_string.strip('` \n').lstrip('json').strip()
                
            report = json.loads(json_string)

        except json.JSONDecodeError:
            logger.error(f"Groq returned non-JSON data, attempting aggressive cleanup...")
            match = re.search(r'\{.*\}', json_string, re.DOTALL)
            if match:
                 json_string_clean = match.group(0)
                 report = json.loads(json_string_clean)
            else:
                raise Exception("Could not find a valid JSON object in AI response.")

        # --- CRITICAL VALIDATION FIX: CHECK for the required nested 'evaluation' key ---
        # The model previously returned flat JSON. We now check the required structure.
        if 'evaluation' not in report or not isinstance(report['evaluation'], list) or len(report['evaluation']) != 5:
            # Attempt to convert flat JSON structure (the problematic output) to the required nested format
            if all(key in report for key in ['Communication', 'Technical Depth', 'Confidence', 'Emotional Control', 'Job Role Alignment']):
                
                logger.warning("Attempting to convert flat JSON response to required nested structure.")
                
                # Reconstruct the required list of dictionaries
                evaluation_list = [
                    {"category": "Communication", "score": report.get('Communication', 0.0), "comment": report.get('Comment', 'N/A')},
                    {"category": "Technical Depth", "score": report.get('Technical Depth', 0.0), "comment": report.get('Comment', 'N/A')},
                    {"category": "Confidence", "score": report.get('Confidence', 0.0), "comment": report.get('Comment', 'N/A')},
                    {"category": "Emotional Control", "score": report.get('Emotional Control', 0.0), "comment": report.get('Comment', 'N/A')},
                    {"category": "Job Role Alignment", "score": report.get('Job Role Alignment', 0.0), "comment": report.get('Comment', 'N/A')},
                ]
                
                # Use the provided Overall Score and Recommendation if they exist, otherwise calculate/generate them
                if 'Overall Score' in report:
                    overall_score = report['Overall Score']
                    recommendation = report['Recommendation']
                else:
                    total_score = sum(item['score'] for item in evaluation_list)
                    overall_score = round(total_score / len(evaluation_list), 1)
                    rec_text = "Recommended for next round" if overall_score >= 6 else "Not recommended at this time"
                    recommendation = f"{rec_text} (Score {overall_score})"
                
                report = {
                    "evaluation": evaluation_list,
                    "overallScore": overall_score,
                    "recommendation": recommendation
                }
                logger.info("Successfully converted flat structure to nested structure for report.")
            else:
                logger.error(f"Invalid JSON structure (KeyError or list length) received from model. Raw report: {report}")
                raise Exception("AI response JSON structure is invalid or incomplete (missing/wrong evaluation list).")


        # Step 3: Re-calculate/Verify to ensure accuracy 
        # (This remains important to catch errors in the model's self-calculated overall score)
        total_score = sum(item['score'] for item in report['evaluation'])
        report['overallScore'] = round(total_score / len(report['evaluation']), 1)
        
        # Step 4: Update recommendation with calculated score
        rec_score = report['overallScore']
        rec_text = "Recommended for next round" if rec_score >= 6 else "Not recommended at this time"
        report['recommendation'] = f"{rec_text} (Score {rec_score})"
        
        return report

    except Exception as e:
        logger.error(f"Failed to generate structured report via Groq (TRIGGERING EXPLICIT ERROR REPORT): {e}")
        # Final Fallback in case of runtime failure (network, API error, JSON parsing error)
        return {
            "evaluation": [
                {"category": "Communication", "score": 0.0, "comment": "API/Network failure: Report generation stopped."},
                {"category": "Technical Depth", "score": 0.0, "comment": "API/Network failure: Report generation stopped."},
                {"category": "Confidence", "score": 0.0, "comment": "API/Network failure: Report generation stopped."},
                {"category": "Emotional Control", "score": 0.0, "comment": "API/Network failure: Report generation stopped."},
                {"category": "Job Role Alignment", "score": 0.0, "comment": "API/Network failure: Report generation stopped."}
            ],
            "overallScore": 0.0,
            "recommendation": "AI Report Generation Failed: Runtime error accessing Groq API."
        }


# --- API Endpoints and Manager (REST OF THE APP) ---

class ConversationManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_states: Dict[str, str] = {}
        self.listening_for_response: Dict[str, bool] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.conversation_states[session_id] = 'idle'
        self.listening_for_response[session_id] = False
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.conversation_states:
            del self.conversation_states[session_id]
        if session_id in self.listening_for_response:
            del self.listening_for_response[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
                logger.info(f"Sent to {session_id}: {message.get('type')}")
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)

manager = ConversationManager()

# API Endpoints
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    logger.info(f"Processing resume upload: {file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        pdf_content = await file.read()

        if len(pdf_content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")

        extracted_text = extract_text_from_pdf(pdf_content)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        parsed_content = parse_resume_content(extracted_text)
        session_id = str(uuid.uuid4())

        sessions[session_id] = {
            "sessionId": session_id,
            "resumeText": extracted_text,
            "resumeParsed": parsed_content,
            "jobDescription": None,
            "questions": [],
            "conversation": [],
            "currentQuestion": 0,
            "status": "resume_uploaded",
            "createdAt": datetime.utcnow(),
            "isActive": False
        }

        return {
            "success": True,
            "sessionId": session_id,
            "extractedText": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "parsedContent": parsed_content,
            "message": "Resume processed successfully"
        }

    except Exception as e:
        logger.error(f"Resume processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(e)}")

@app.post("/setup-interview")
async def setup_interview(request: InterviewRequest):
    try:
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = sessions[request.session_id]

        job_desc_text = request.job_description.text if request.job_description else ""
        job_role = request.job_description.role if request.job_description else "General Technical Role"
        
        session_data["resumeParsed"]["jobDescription"] = {"role": job_role}
        
        questions = generate_interview_questions(
            session_data["resumeText"], 
            job_desc_text,
            session_data["resumeParsed"]
        )

        sessions[request.session_id].update({
            "questions": questions,
            "jobDescription": {
                "text": job_desc_text,
                "role": job_role
            },
            "status": "ready_for_interview",
        })

        return {
            "success": True,
            "questions": questions,
            "status": "ready_for_interview",
            "message": f"Generated {len(questions)} conversational questions"
        }

    except Exception as e:
        logger.error(f"Setup interview error: {e}")
        raise HTTPException(status_code=500, detail="Failed to setup interview")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for conversational AI interview - AI asks questions AND listens to answers"""
    await manager.connect(websocket, session_id)

    try:
        if session_id not in sessions:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "Session not found"
            }))
            return

        session_data = sessions[session_id]
        questions = session_data.get("questions", [])

        if not questions:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No questions available. Please setup interview first."
            }))
            return

        # Start conversation
        session_data["isActive"] = True
        session_data["startTime"] = datetime.utcnow()
        manager.conversation_states[session_id] = 'asking'

        await asyncio.sleep(1)

        if questions and session_data["isActive"]:
            first_question = questions[0]

            await manager.send_message(session_id, {
                "type": "ai_question",
                "content": first_question["question"],
                "questionId": first_question["id"],
                "questionNumber": 1,
                "totalQuestions": len(questions),
                "category": first_question["category"],
                "speak": True,
                "state": "ai_speaking"
            })

            session_data["conversation"].append({
                "role": "ai",
                "content": first_question["question"],
                "timestamp": datetime.utcnow().isoformat(),
                "questionId": first_question["id"]
            })

            session_data["currentQuestion"] = 0

        # Handle conversation loop
        while session_data.get("isActive", False):
            try:
                data = await websocket.receive_text()
                data_json = json.loads(data)
                message_type = data_json.get("type")

                logger.info(f"Received: {message_type} (State: {manager.conversation_states.get(session_id)})")
                
                # --- NEW MESSAGE TYPE FOR TTS COMPLETED ---
                if message_type == "speech_finished":
                    # This is the signal from the client that the question/response has been fully read.
                    if manager.conversation_states.get(session_id) in ('asking', 'responding'):
                        manager.conversation_states[session_id] = 'listening'
                        manager.listening_for_response[session_id] = True
                        
                        await manager.send_message(session_id, {
                            "type": "start_listening",
                            "message": "I'm listening for your response...",
                            "state": "listening_for_answer"
                        })
                    continue 

                if message_type == "audio_response" and manager.listening_for_response.get(session_id, False):
                    # Process user's audio response
                    manager.conversation_states[session_id] = 'processing'
                    manager.listening_for_response[session_id] = False

                    await manager.send_message(session_id, {
                        "type": "processing_response",
                        "message": "Processing your response...",
                        "state": "processing_answer"
                    })

                    audio_content = data_json.get("content", "")

                    if audio_content:
                        try:
                            audio_bytes = base64.b64decode(audio_content)

                            # Transcribe user response
                            transcription = await transcribe_audio_with_groq(audio_bytes)

                            if transcription and len(transcription.strip()) > 5:
                                # Show user response in chat
                                await manager.send_message(session_id, {
                                    "type": "user_response_shown",
                                    "content": transcription,
                                    "state": "user_answered"
                                })

                                # Add to conversation
                                session_data["conversation"].append({
                                    "role": "user",
                                    "content": transcription,
                                    "timestamp": datetime.utcnow().isoformat()
                                })

                                # Generate AI conversational response/action
                                current_q_index = session_data.get("currentQuestion", 0)
                                current_question = questions[current_q_index] if current_q_index < len(questions) else {}

                                context = {
                                    "current_question": current_question,
                                    "conversation": session_data["conversation"],
                                    "session_data": session_data 
                                }

                                ai_output = await generate_conversational_response(transcription, context)
                                ai_response = ai_output["response"]
                                ai_action = ai_output["action"]
                                
                                manager.conversation_states[session_id] = 'responding'

                                await manager.send_message(session_id, {
                                    "type": "ai_conversational_response",
                                    "content": ai_response,
                                    "speak": True,
                                    "state": "ai_responding"
                                })

                                session_data["conversation"].append({
                                    "role": "ai", 
                                    "content": ai_response,
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                
                                # CRITICAL FLOW CONTROL: Check if AI has signaled to move on
                                if ai_action == ACTION_NEXT_Q:
                                    # The AI has gracefully concluded the current topic. Trigger the next question logic.
                                    
                                    # If the next question index is out of bounds, complete the interview
                                    next_q_index = current_q_index + 1
                                    
                                    if next_q_index < len(questions):
                                        # Manually send the next_question type to simulate a button click/advancement
                                        manager.conversation_states[session_id] = 'waiting_to_ask'
                                        await asyncio.sleep(0.5) # Wait half a second for previous TTS to finish buffering
                                        await manager.send_message(session_id, {"type": "next_question"}) 
                                        
                                    else:
                                        # The last question was answered, and the AI asked to conclude.
                                        # Trigger the final report generation logic immediately.
                                        await manager.send_message(session_id, {"type": "end_interview"})
                                        break # Exit the loop, report generation handled by end_interview logic

                            else:
                                # No valid transcription - ask to repeat
                                await manager.send_message(session_id, {
                                    "type": "ask_repeat",
                                    "content": "I'm sorry, I didn't catch that clearly. Could you please repeat your answer?",
                                    "speak": True
                                })

                        except Exception as e:
                            logger.error(f"Error processing audio response: {e}")

                            await manager.send_message(session_id, {
                                "type": "processing_error",
                                "content": "I had trouble processing your response. Could you please try again?",
                                "speak": True
                            })

                elif message_type == "next_question":
                    # Move to next structured question (Triggered by client or by AI action above)
                    current_q_index = session_data.get("currentQuestion", 0)
                    next_q_index = current_q_index + 1

                    manager.listening_for_response[session_id] = False
                    manager.conversation_states[session_id] = 'asking'

                    if next_q_index < len(questions):
                        next_question = questions[next_q_index]
                        session_data["currentQuestion"] = next_q_index

                        await manager.send_message(session_id, {
                            "type": "ai_question",
                            "content": next_question["question"],
                            "questionId": next_question["id"],
                            "questionNumber": next_q_index + 1,
                            "totalQuestions": len(questions),
                            "category": next_question["category"],
                            "speak": True,
                            "state": "ai_asking_next"
                        })

                        session_data["conversation"].append({
                            "role": "ai",
                            "content": next_question["question"],
                            "timestamp": datetime.utcnow().isoformat(),
                            "questionId": next_question["id"]
                        })

                    else:
                        # Interview completed (No more structured questions)
                        # This handles the case where the client clicked 'next' on the final question.
                        
                        report_data = await generate_interview_report(session_data)
                        
                        session_data["status"] = "completed"
                        session_data["endTime"] = datetime.utcnow()

                        completion_msg = "Thank you so much for this wonderful conversation! You've shared some really insightful responses. That concludes our interview."

                        await manager.send_message(session_id, {
                            "type": "interview_completed",
                            "content": completion_msg,
                            "speak": True,
                            "state": "interview_finished",
                            "report": report_data 
                        })

                        session_data["conversation"].append({
                            "role": "ai",
                            "content": completion_msg,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                        break

                elif message_type == "end_interview":
                    # Interview manually ended (or triggered by final AI action)
                    
                    report_data = await generate_interview_report(session_data)
                    
                    session_data["status"] = "completed"
                    session_data["endTime"] = datetime.utcnow()

                    end_msg = "Thank you for your time! It was great talking with you. Your final report is ready."

                    await manager.send_message(session_id, {
                        "type": "interview_completed", 
                        "content": end_msg,
                        "speak": True,
                        "state": "interview_ended",
                        "report": report_data 
                    })

                    break

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket conversation error: {e}")
                # Log traceback for better debugging
                logger.error(traceback.format_exc())

    finally:
        if session_id in sessions:
            sessions[session_id]["isActive"] = False
        manager.disconnect(session_id)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "9.3.1",
        "features": {
            "conversationalInterview": True,
            "aiAsksAndListens": True,
            "realTimeConversation": True,
            "groqEnabled": bool(groq_client)
        },
        "activeSessions": len([s for s in sessions.values() if s.get("isActive", False)])
    }

if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable provided by the hosting platform, or default to 8000
    PORT = int(os.getenv("PORT", 8000))
    print("🎤 Starting Conversational AI Interview System...")
    print("📍 AI asks questions AND listens to your answers")
    print("💬 Real back-and-forth conversation like a human interviewer")
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=DEBUG)
