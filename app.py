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
import re # Added for pattern matching

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

# Initialize FastAPI app
app = FastAPI(
    title="Conversational AI Interview System",
    description="AI asks questions AND listens to your answers in real conversation",
    version="8.6.0" # Updated version for improved flow, aggressive pivot, and graceful conclusion
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

# Initialize Groq client
groq_client = None
if GROQ_AVAILABLE and GROQ_API_KEY and len(GROQ_API_KEY) > 10:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq client initialized - Real AI conversation enabled")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        groq_client = None
else:
    logger.info("ℹ️ Running with enhanced simulation mode")

# Models
class JobDescription(BaseModel):
    text: str
    role: str

class InterviewRequest(BaseModel):
    session_id: str
    job_description: Optional[JobDescription] = None

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

    # --- 1. Attempt to extract Name ---
    candidate_name = ""
    if lines:
        for line in lines[:5]:
            # Simple heuristic: title-cased line, likely the name, not too long
            if line == line.title() and 1 < len(line.split()) < 4 and len(line) > 4 and len(line) < 40:
                # Exclude common headers like 'Education'
                if not any(keyword in line.lower() for keyword in ['skill', 'education', 'experience', 'contact']):
                    candidate_name = line
                    break
    
    if not candidate_name:
         candidate_name = "Candidate"
        
    # --- 2. Extract Skills and Experience ---
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

    for line in lines:
        lower_line = line.lower()

        # Find skills
        for skill in skill_keywords:
            if re.search(r'\b' + re.escape(skill.split()[0]) + r'\b', lower_line):
                found_skills.add(skill)

        # Find experience lines (job titles/descriptions)
        if any(exp in lower_line for exp in ['developer', 'engineer', 'manager', 'analyst', 'architect', 'lead']) and len(line) > 20:
            experience_lines.append(line)
            
    # Clean up and prioritize the most relevant skills
    clean_skills = sorted(list(set([s.title() for s in found_skills])))
    
    return {
        "candidateName": candidate_name, # Added candidate name
        "skills": clean_skills[:15],
        "experience": experience_lines[:8],
        "education": []
    }
    

def generate_interview_questions(resume_text: str, job_description: str = None, resume_parsed: Dict = None) -> List[Dict]:
    """
    Generates a list of interview questions with a priority on introduction,
    project deep dive, JD alignment, and concept deep dive. Questions are personalized to resume content.
    """
    questions = []
    resume_lower = resume_text.lower() if resume_text else ""
    is_technical = any(tech in resume_lower for tech in ['developer', 'engineer', 'programmer', 'python', 'javascript', 'machine learning'])
    
    # Context variables
    candidate_name = resume_parsed.get('candidateName', 'Candidate')
    main_skill = resume_parsed['skills'][0] if resume_parsed['skills'] else 'core technical areas'
    # NOTE: Removing dependency on first_experience string to prevent verbose prompt repetition
    
    # --- 1. Introduction (Must be first, personalized and focused) ---
    questions.append({
        "id": 1,
        "category": "Introduction", 
        "question": f"Welcome, **{candidate_name}**! I'm your AI interviewer, and I'm looking forward to diving into your background. Could you begin by walking me through your professional journey, starting with your most recent or significant role?",
        "follow_ups": [
            "What is your proudest professional achievement mentioned in your background?",
            "How does your career trajectory align with your long-term goals?"
        ]
    })

    # --- 2. Project Deep Dive (High Priority, targets key skill) ---
    questions.append({
        "id": 2,
        "category": "Project Deep Dive",
        "question": f"Given your background, let's dive deep. Can you walk me through the most complex project you built that utilizes **{main_skill}**, focusing on the system's architecture and the most critical technical trade-off you faced?",
        "follow_ups": [
            "What was the most significant technical trade-off you had to make, and why?",
            "If you had infinite resources, how would you rebuild that system for maximum scalability?",
            "What were the key architectural decisions, and how did they impact the final product?"
        ]
    })
    
    # --- 3. Job Description Alignment (If provided, ensures JD focus) ---
    if job_description and len(job_description.strip()) > 50:
          job_role = resume_parsed.get('jobDescription', {}).get('role', 'the technical role')
          questions.append({
            "id": 3,
            "category": "JD Alignment",
            "question": f"Based on the **{job_role}** you're applying for, which specific skills or experiences from your background do you feel make you the strongest candidate to address the core technical challenges of the role?",
            "follow_ups": [
                "Could you provide a detailed example of when you demonstrated the skill of X (e.g. teamwork, leadership)?",
                "What do you perceive as the biggest challenge in this role, and how would you tackle it?"
            ]
        })
    else:
        # If no JD, focus this question on general motivation and career aspirations based on skills
        questions.append({
            "id": 3,
            "category": "Motivation",
            "question": "Looking at your skills, what is the most challenging technical (or professional) hurdle you've overcome, and what did you learn from it that you apply to new projects?",
            "follow_ups": [
                "How do you handle situations where you receive critical feedback on your work?",
                "What is one area of your professional skill set that you are actively trying to improve?"
            ]
        })


    # --- 4. Concept Deep Dive (For Technical/Conceptually Deep Questions) ---
    if is_technical and resume_parsed.get("skills"):
        main_skill_concept = next((s for s in ['Python', 'Java', 'SQL', 'React', 'AWS'] if s in resume_parsed['skills']), 'a core technical concept')
        questions.append({
            "id": 4,
            "category": "Concept Deep Dive",
            "question": f"Let's test your fundamental knowledge in **{main_skill_concept}**. If you were debugging a performance issue in a high-traffic system, describe the exact tools and methodology you would use to pinpoint the bottleneck.",
            "follow_ups": [
                f"Can you explain the difference between L1 and L2 regularization and when you would choose one over the other?",
                "Describe a situation where you had to debug a race condition in a production environment.",
                "How do you implement and measure true performance optimization in a large-scale system?",
                f"Beyond the basics, what are the most critical, yet often misunderstood, aspects of {main_skill_concept}?"
            ]
        })
    
    # --- 5. General and Closing ---
    questions.append({
        "id": len(questions) + 1,
        "category": "Closing",
        "question": "We've covered a lot of ground today. Do you have any questions for me about the role, the team, or the company?",
        "follow_ups": [
            "If you were hired, what would you hope to accomplish in your first 90 days?",
        ]
    })

    return questions

async def transcribe_audio_with_groq(audio_data: bytes) -> str:
    """Transcribe audio using Groq Whisper"""
    if not audio_data or len(audio_data) < 1000:
        return ""

    if not groq_client:
        # Enhanced simulation responses
        responses = [
            "I have about 3 years of experience as a software developer, mainly working on web applications using Python and JavaScript.",
            "My most challenging project was building a real-time dashboard that had to handle thousands of concurrent users. I used WebSockets and Redis for caching.",
            "I really enjoy problem-solving and creating solutions that make people's work easier. I'm passionate about writing clean, maintainable code.",
            "When debugging, I usually start by reproducing the issue, then I use logging and debugging tools to trace through the code step by step.",
            "I stay current by following tech blogs, taking online courses, and working on side projects to experiment with new technologies."
        ]
        import random
        return random.choice(responses)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            temp_filename = temp_file.name

        try:
            with open(temp_filename, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
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

async def generate_conversational_response(user_text: str, context: Dict) -> str:
    """Generate conversational AI response that continues the interview, prioritizing a deep drill-down."""
    if not user_text or len(user_text.strip()) < 5:
        return "I'm sorry, I didn't catch that. Could you please repeat your response?"

    current_question = context.get('current_question', {})
    question_category = current_question.get('category', 'General')
    conversation_history = context.get('conversation', [])
    session_data = context.get('session_data', {})
    resume_parsed = session_data.get('resumeParsed', {})
    
    # --- Enhanced Simulation Mode Responses (Updated) ---
    if not groq_client:
        # Simulation responses adapted to be friendly and drill down
        candidate_name = resume_parsed.get('candidateName', 'Candidate')
        
        category_responses = {
            'Introduction': [
                f"That's a great overview, {candidate_name}! I'm keen to hear more about your technical competencies now.",
                f"Thank you for that introduction, {candidate_name}. It sounds like you're passionate about Machine Learning."
            ],
            'Project Deep Dive': [
                f"That sounds like a complex project, {candidate_name}! To drill down on the architecture: what specific trade-offs did you make regarding latency versus throughput?",
                f"Impressive work, {candidate_name}! Can you explain the *exact* data structure you chose to handle the memory constraints when processing large, real-time datasets?"
            ],
            'Concept Deep Dive': [
                f"That's a solid explanation, {candidate_name}. Now, let's take that a layer deeper: why is it generally recommended to normalize the features before training a linear regression model, and what happens if you skip that step?",
                f"Very insightful, {candidate_name}. Based on your experience, how does **Python's GIL** affect high-performance data processing, and what *specific* library or approach do you use to work around it?"
            ],
            'JD Alignment': [
                f"That direct connection to the role is helpful, {candidate_name}. Tell me more about a time you led a data cleaning effort, focusing on your choice of tools.",
                f"Your skills align well, {candidate_name}! What part of our company's mission resonates most with your Data Science goals?"
            ],
            'General': [
                f"That's very insightful, {candidate_name}. Can you tell me more about how that experience shaped your approach?",
                f"Interesting perspective, {candidate_name}! What did you learn from that situation?"
            ]
        }
        
        # Check if user requested a pivot or transition (Simulation)
        user_lower = user_text.lower()
        if any(phrase in user_lower for phrase in ["can't answer further", "proceed for further questions", "move on", "next question"]):
             return f"I understand, {candidate_name}. Let's conclude this topic and move to the next structured question."
        
        # Simulation: Immediate agreement to pivot to Java coding
        if "java coding questions" in user_lower or ("java" in user_lower and "coding" in user_lower and not "oops" in user_lower) :
            return f"Absolutely, {candidate_name}. Let's pivot to Java coding. Describe how you would implement a thread-safe singleton pattern in Java 8 or later."
        
        # Check for technical drill-down opportunity
        technical_keywords = ['python', 'java', 'oops', 'ml', 'machine learning', 'data science', 'aws', 'docker']
        for skill in technical_keywords:
             if skill in user_lower:
                return f"Excellent, {candidate_name}! You mentioned **{skill}**. Let's dive into that: can you explain the difference between **shallow and deep copies** in Python and provide a scenario where a shallow copy caused a significant production bug?"
        
        # Simple meta-query handling simulation
        if any(phrase in user_lower for phrase in ["time is it", "how many questions", "company name"]):
            if "time is it" in user_lower:
                return f"It's currently {datetime.now().strftime('%H:%M')}. Let's get back to your technical experience now, {candidate_name}."
            elif "how many questions" in user_lower:
                total_q = len(session_data.get("questions", []))
                current_q_index = session_data.get("currentQuestion", 0)
                return f"We're on question {current_q_index + 1} out of {total_q}, {candidate_name}. Please continue with your previous answer."
            else:
                return f"The company name is fictitious for this simulation, {candidate_name}. Now, regarding your project..."


        responses = category_responses.get(question_category, category_responses['General'])
        import random
        return random.choice(responses)
    # --- End Simulation Mode ---

    try:
        # Build conversation context
        recent_context = ""
        if conversation_history:
            recent_exchanges = conversation_history[-4:]  # Last 2 Q&A pairs
            for entry in recent_exchanges:
                role = "Interviewer" if entry.get("role") == "ai" else "Candidate"
                # Limit content length for system prompt efficiency
                content_preview = entry.get('content', '')[:100].replace('\n', ' ') 
                recent_context += f"{role}: {content_preview}...\n"
        
        # Additional context for deep dives
        resume_context = json.dumps(resume_parsed.get('skills', []))
        candidate_name = resume_parsed.get('candidateName', 'Candidate')
        job_role = session_data.get('jobDescription', {}).get('role', 'a technical role')

        # --- CRITICAL SYSTEM PROMPT REVISION FOR TECHNICAL DRILL DOWN ---
        system_prompt = f"""You are a highly skilled, **friendly, encouraging, and professional** AI interviewer specializing in **{job_role}**. Your primary goal is to conduct a **deep-dive conversation** to rigorously assess the candidate's core technical and conceptual knowledge, all while maintaining an encouraging tone.

Candidate: **{candidate_name}**
Current Interview Focus: **{question_category}**
Candidate's Key Skills (from resume, to guide deep-drills): {resume_context}

Recent conversation:
{recent_context}

The candidate just responded: "{user_text}"

Your next step is critical. Your instructions are:
1.  **Tone and Acknowledgment:** Start by giving a friendly acknowledgment that **must include the candidate's name, {candidate_name}**, to establish a professional and encouraging rapport (e.g., "Excellent point, {candidate_name}!" or "That's insightful, {candidate_name}."). **Do not use confrontational language (e.g., 'bold assertion').**
2.  **CRITICAL PIVOT/TRANSITION HANDLING (Highest Priority):**
    * **DOMAIN SWITCH:** If the candidate explicitly requests a specific new topic, language, or topic change (e.g., 'Java questions', 'Python coding questions', 'Leave OOP', 'move to coding'), **immediately agree** and ask a challenging question specific to that new preference. Do not argue, negotiate, or revert to the previous topic/language.
    * **INTERVIEW END:** If the candidate says 'let's conclude,' 'I'm done,' or 'sorry, I'm going,' generate a single, polite sentence to conclude the interview and signal the end (e.g., "I understand. Thank you for your time, {candidate_name}. That concludes our interview.").
3.  **META-QUERY HANDLING (General Queries):**
    * If the candidate's response is a general query (e.g., 'What time is it?'), **answer concisely** and then immediately prompt them back to the original technical question.
4.  **TECHNICAL DEEP DRILL RULE (Standard Follow-up):**
    * If the candidate *is* answering the question and has *not* requested a pivot or end:
        * **DRILL DOWN:** Ask a challenging follow-up that seeks *specific implementation details*, *rationale for trade-offs*, or *metrics* based directly on their previous answer. **Avoid simple behavioral questions.**

**Length Constraint:** Keep your entire response to **one or two concise, encouraging sentences (under 150 characters total)**. Maintain professional rigor throughout."""
        # --- END SYSTEM PROMPT REVISION ---

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.85, # Increased temperature slightly for more natural conversation
            max_tokens=150
        )

        ai_response = response.choices[0].message.content.strip()
        logger.info(f"Generated conversational response: {ai_response[:80]}...")
        return ai_response

    except Exception as e:
        logger.error(f"AI response generation failed: {e}")
        return "That's really insightful! I'd love to follow up on that point to understand your thinking better."

# --- API Endpoints and Manager (Remaining code is the same) ---

# WebSocket Connection Manager (same)
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
        
        # Update job role in parsed data for better question generation
        session_data["resumeParsed"]["jobDescription"] = {"role": job_role}
        
        # Pass parsed content to question generation
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

        # Fetch candidate name for personalized intro
        candidate_name = session_data["resumeParsed"].get("candidateName", "Candidate")
        
        # --- FIX: Removed the redundant initial 'ai_message' (welcome_msg) ---
        # The intro is now merged into the first 'ai_question' below, improving flow.
        
        # Wait and ask first question
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

            # Now wait for user response
            await asyncio.sleep(2)  # Give time for AI to finish speaking

            manager.conversation_states[session_id] = 'listening'
            manager.listening_for_response[session_id] = True

            await manager.send_message(session_id, {
                "type": "start_listening",
                "message": "I'm listening for your response...",
                "state": "listening_for_answer"
            })

        # Handle conversation loop
        while session_data.get("isActive", False):
            try:
                data = await websocket.receive_text()
                data_json = json.loads(data)
                message_type = data_json.get("type")

                logger.info(f"Received: {message_type} (State: {manager.conversation_states.get(session_id)})")

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

                                # Generate AI conversational response
                                current_q_index = session_data.get("currentQuestion", 0)
                                current_question = questions[current_q_index] if current_q_index < len(questions) else {}

                                context = {
                                    "current_question": current_question,
                                    "conversation": session_data["conversation"],
                                    "session_data": session_data # Pass all session data for rich context
                                }

                                ai_response = await generate_conversational_response(transcription, context)

                                # Send AI conversational response
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

                                # Wait for AI to finish speaking, then continue
                                await asyncio.sleep(4)
                                
                                # Check if AI response contained a definitive transition phrase to move to next structured Q
                                transition_phrases = ["let's move to the next question", "concludes this topic", "I understand, we can move"]
                                needs_manual_next = not any(phrase in ai_response.lower() for phrase in transition_phrases)


                                if session_data.get("isActive", False):
                                    # Resume listening for more from user or move to next question
                                    manager.conversation_states[session_id] = 'listening'
                                    manager.listening_for_response[session_id] = True

                                    await manager.send_message(session_id, {
                                        "type": "continue_listening",
                                        "message": "Please continue, or I can move to the next question...",
                                        "state": "listening_for_more"
                                    })

                            else:
                                # No valid transcription - ask to repeat
                                await manager.send_message(session_id, {
                                    "type": "ask_repeat",
                                    "content": "I'm sorry, I didn't catch that clearly. Could you please repeat your answer?",
                                    "speak": True,
                                    "state": "asking_repeat"
                                })

                                await asyncio.sleep(2)

                                manager.conversation_states[session_id] = 'listening'
                                manager.listening_for_response[session_id] = True

                                await manager.send_message(session_id, {
                                    "type": "resume_listening",
                                    "message": "I'm listening...",
                                    "state": "listening_for_answer"
                                })

                        except Exception as e:
                            logger.error(f"Error processing audio response: {e}")

                            await manager.send_message(session_id, {
                                "type": "processing_error",
                                "content": "I had trouble processing your response. Could you please try again?",
                                "speak": True
                            })

                            manager.conversation_states[session_id] = 'listening'
                            manager.listening_for_response[session_id] = True

                elif message_type == "next_question":
                    # Move to next question
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

                        # Wait then start listening for response
                        await asyncio.sleep(3)

                        manager.conversation_states[session_id] = 'listening'
                        manager.listening_for_response[session_id] = True

                        await manager.send_message(session_id, {
                            "type": "start_listening",
                            "message": "I'm listening for your response to this question...",
                            "state": "listening_for_answer"
                        })

                    else:
                        # Interview completed
                        session_data["status"] = "completed"
                        session_data["endTime"] = datetime.utcnow()

                        completion_msg = "Thank you so much for this wonderful conversation! You've shared some really insightful responses. That concludes our interview."

                        await manager.send_message(session_id, {
                            "type": "interview_completed",
                            "content": completion_msg,
                            "speak": True,
                            "state": "interview_finished"
                        })

                        session_data["conversation"].append({
                            "role": "ai",
                            "content": completion_msg,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                elif message_type == "end_interview":
                    session_data["status"] = "completed"
                    session_data["endTime"] = datetime.utcnow()

                    end_msg = "Thank you for your time! It was great talking with you."

                    await manager.send_message(session_id, {
                        "type": "interview_completed",
                        "content": end_msg,
                        "speak": True,
                        "state": "interview_ended"
                    })

                    break

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket conversation error: {e}")

    finally:
        if session_id in sessions:
            sessions[session_id]["isActive"] = False
        manager.disconnect(session_id)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "8.6.0",
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

