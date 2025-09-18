# Updated Flask API Server - With .env Configuration
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import random
import datetime
from typing import List, Dict, Any, Optional
import joblib
import requests
import time
import uuid
import os
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')

CORS(app)

class RAGExcelInterviewAPI:
    """
    Flask API adaptation with .env configuration
    """
    
    def __init__(self, 
                 model_path="sessionmanager/ridge_multioutput_regressor_updated_model.joblib",
                 vectorizer_path="sessionmanager/tfidf_vectorizer_updated_model.joblib",
                 dataset_path="sessionmanager/excel_qa_augmented_dynamic_with_scores_structured.json",
                 results_directory="interview_results"):
        
        # Load ML components
        try:
            self.scoring_model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("✅ AI Scoring Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not load AI model ({e}). Using fallback scoring.")
            self.scoring_model = None
            self.vectorizer = None
            
        # Load dataset
        try:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)
            print(f"✅ Dataset loaded: {len(self.dataset)} Q&A pairs")
        except Exception as e:
            print(f"⚠️ Warning: Could not load dataset ({e}). Using sample questions.")
            self.dataset = []
            
        # Results storage setup
        self.results_directory = results_directory
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
            print(f"✅ Created results directory: {results_directory}")
        
        # API configuration from .env file
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.api_endpoint = os.getenv('PERPLEXITY_API_ENDPOINT', 'https://api.perplexity.ai/chat/completions')
        
        # Validate environment variables
        if not self.api_key:
            print("❌ ERROR: PERPLEXITY_API_KEY not found in .env file!")
            print("🔧 Please create a .env file one directory up with PERPLEXITY_API_KEY=your_key_here")
        else:
            print(f"✅ API Key loaded from .env: {self.api_key[:10]}...")
            print(f"✅ API Endpoint: {self.api_endpoint}")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Round configurations (same as original)
        self.round_configs = {
            1: {
                "name": "Foundation Round",
                "description": "Basic Excel knowledge and fundamental operations",
                "question_types": ["Beginner"],
                "num_questions": 3,
                "time_limit_minutes": 15
            },
            2: {
                "name": "Intermediate Round",
                "description": "Formulas, functions, and data manipulation",
                "question_types": ["Intermediate"],
                "num_questions": 3,
                "time_limit_minutes": 20
            },
            3: {
                "name": "Advanced Round", 
                "description": "Complex analysis, pivot tables, and advanced functions",
                "question_types": ["Advanced"],
                "num_questions": 2,
                "time_limit_minutes": 25
            }
        }
        
        # Active sessions storage (in production, use Redis/Database)
        self.active_sessions = {}
    
    def call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Call LLM using environment variables"""
        if not self.api_key:
            print("⚠️ No API key available, using fallback response")
            return self._get_fallback_response(prompt)
            
        try:
            data = {
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "You are an AI Excel Interview Specialist conducting a live interview. Speak directly to the candidate using 'you' and 'your'. Be encouraging, professional, and conversational. Act as their interviewer, not a reviewer."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens
            }

            response = requests.post(self.api_endpoint, headers=self.headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return self._get_fallback_response(prompt)
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._get_fallback_response(prompt)
    
    def _get_fallback_response(self, prompt: str) -> str:
        """Fallback responses when API is unavailable"""
        if "introduce" in prompt.lower() or "specialist" in prompt.lower():
            return """Hello! I'm your AI Excel Interview Specialist. I'm excited to evaluate your Excel skills today!

We'll go through 3 rounds:
• Foundation Round: 3 questions on basic Excel knowledge
• Intermediate Round: 3 questions on formulas and data manipulation  
• Advanced Round: 2 questions on complex analysis

I'll score each response on Correctness, Clarity, Terminology, and Efficiency, giving you immediate feedback after each question.

Are you ready to begin? Let's start with some foundational Excel questions!"""
        
        elif "round" in prompt.lower():
            return "Great! Let's move to the next part of our interview. I'm excited to see your Excel knowledge in action."
        
        elif "feedback" in prompt.lower():
            return "Thank you for your detailed answer! You've shown good understanding of Excel concepts. I'd encourage you to include more specific examples in future responses to demonstrate deeper expertise."
        
        elif "interview complete" in prompt.lower() or "final" in prompt.lower():
            return """Excellent work completing the Excel skills assessment! 

Based on your responses, you've demonstrated solid foundational knowledge of Excel with clear explanations and good conceptual understanding.

Key Strengths:
• Clear communication of Excel concepts
• Good understanding of basic functionality

Growth Areas:
• Practice with more advanced formulas and functions
• Working with larger, complex datasets

Next Steps: Continue building your Excel skills through hands-on practice and exploring advanced features like Power Query and advanced charting.

Overall Assessment: You show strong potential as an intermediate Excel user. Keep practicing and you'll reach advanced proficiency soon!"""
        
        else:
            return "Thank you for your response. I appreciate your effort and encourage you to keep developing your Excel skills!"
    
    def save_interview_results(self, session_data: dict, final_summary: str, avg_scores: dict) -> str:
        """Save complete interview results to file"""
        try:
            # Create comprehensive results object
            interview_results = {
                "session_info": {
                    "session_id": session_data["session_id"],
                    "start_time": session_data["start_time"].isoformat() if isinstance(session_data["start_time"], datetime.datetime) else session_data["start_time"],
                    "end_time": datetime.datetime.now().isoformat(),
                    "total_duration_minutes": round((datetime.datetime.now() - session_data["start_time"]).total_seconds() / 60, 2) if isinstance(session_data["start_time"], datetime.datetime) else "N/A"
                },
                "summary": {
                    "total_questions": len(session_data["scores"]),
                    "rounds_completed": session_data.get("current_round", 1),
                    "average_scores": avg_scores,
                    "overall_performance": self._categorize_performance(avg_scores["overall"]),
                    "ai_summary": final_summary
                },
                "detailed_responses": session_data["responses"],
                "round_breakdown": self._calculate_round_breakdown(session_data),
                "performance_metrics": {
                    "highest_scoring_area": max(avg_scores, key=avg_scores.get),
                    "lowest_scoring_area": min(avg_scores, key=avg_scores.get),
                    "consistency_score": self._calculate_consistency(session_data["scores"]),
                    "improvement_trend": self._analyze_improvement_trend(session_data["scores"])
                }
            }
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interview_{session_data['session_id'][:8]}_{timestamp}.json"
            filepath = os.path.join(self.results_directory, filename)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(interview_results, f, indent=2, default=str)
            
            print(f"✅ Interview results saved to: {filepath}")
            
            # Also save a summary CSV for easy analysis
            self._save_summary_csv(interview_results)
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            return None
    
    def _categorize_performance(self, overall_score: float) -> str:
        """Categorize overall performance based on score"""
        if overall_score >= 90:
            return "Excellent"
        elif overall_score >= 80:
            return "Good"
        elif overall_score >= 70:
            return "Fair"
        elif overall_score >= 60:
            return "Developing"
        else:
            return "Needs Improvement"
    
    def _calculate_round_breakdown(self, session_data: dict) -> dict:
        """Calculate performance breakdown by round"""
        round_breakdown = {}
        responses = session_data.get("responses", [])
        
        current_round = 1
        round_responses = []
        
        for i, response in enumerate(responses):
            # Determine which round this response belongs to
            if current_round == 1 and i >= 3:  # Foundation round has 3 questions
                if round_responses:
                    round_breakdown[f"Round {current_round}"] = self._calculate_round_average(round_responses)
                current_round = 2
                round_responses = []
            elif current_round == 2 and i >= 6:  # Intermediate round questions 4-6
                if round_responses:
                    round_breakdown[f"Round {current_round}"] = self._calculate_round_average(round_responses)
                current_round = 3
                round_responses = []
            
            round_responses.append(response)
        
        # Add the last round
        if round_responses:
            round_breakdown[f"Round {current_round}"] = self._calculate_round_average(round_responses)
        
        return round_breakdown
    
    def _calculate_round_average(self, round_responses: list) -> dict:
        """Calculate average scores for a round"""
        if not round_responses:
            return {}
        
        total_scores = {"correctness": 0, "clarity": 0, "terminology": 0, "efficiency": 0, "overall": 0}
        count = len(round_responses)
        
        for response in round_responses:
            scores = response.get("scores", {})
            for key in total_scores:
                total_scores[key] += scores.get(key, 0)
        
        return {key: round(value / count, 1) for key, value in total_scores.items()}
    
    def _calculate_consistency(self, scores: list) -> float:
        """Calculate consistency score based on standard deviation"""
        if not scores or len(scores) < 2:
            return 100.0
        
        overall_scores = [score["overall"] for score in scores]
        mean_score = sum(overall_scores) / len(overall_scores)
        variance = sum((score - mean_score) ** 2 for score in overall_scores) / len(overall_scores)
        std_dev = variance ** 0.5
        
        # Convert to consistency score (lower std_dev = higher consistency)
        consistency = max(0, 100 - (std_dev * 2))  # Scale factor of 2
        return round(consistency, 1)
    
    def _analyze_improvement_trend(self, scores: list) -> str:
        """Analyze if performance improved over time"""
        if not scores or len(scores) < 3:
            return "Insufficient data"
        
        overall_scores = [score["overall"] for score in scores]
        first_half = overall_scores[:len(overall_scores)//2]
        second_half = overall_scores[len(overall_scores)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        diff = second_avg - first_avg
        
        if diff > 5:
            return "Improving"
        elif diff < -5:
            return "Declining"
        else:
            return "Stable"
    
    def _save_summary_csv(self, interview_results: dict):
        """Save a summary CSV for easy analysis"""
        try:
            import csv
            
            csv_filename = "interview_summaries.csv"
            csv_filepath = os.path.join(self.results_directory, csv_filename)
            
            # Check if file exists to write headers
            file_exists = os.path.exists(csv_filepath)
            
            with open(csv_filepath, 'a', newline='') as csvfile:
                fieldnames = [
                    'session_id', 'date', 'duration_minutes', 'total_questions', 
                    'overall_score', 'correctness', 'clarity', 'terminology', 'efficiency',
                    'performance_category', 'consistency_score', 'improvement_trend'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                # Extract data for CSV
                session_info = interview_results["session_info"]
                avg_scores = interview_results["summary"]["average_scores"]
                metrics = interview_results["performance_metrics"]
                
                writer.writerow({
                    'session_id': session_info["session_id"][:8],
                    'date': session_info["start_time"][:10],
                    'duration_minutes': session_info["total_duration_minutes"],
                    'total_questions': interview_results["summary"]["total_questions"],
                    'overall_score': avg_scores["overall"],
                    'correctness': avg_scores["correctness"],
                    'clarity': avg_scores["clarity"],
                    'terminology': avg_scores["terminology"],
                    'efficiency': avg_scores["efficiency"],
                    'performance_category': interview_results["summary"]["overall_performance"],
                    'consistency_score': metrics["consistency_score"],
                    'improvement_trend': metrics["improvement_trend"]
                })
                
            print(f"✅ Summary added to CSV: {csv_filepath}")
            
        except Exception as e:
            print(f"⚠️ Could not save CSV summary: {str(e)}")
    
    def _retrieve_question_for_round(self, round_num):
        """Retrieve question from dataset - same as original"""
        if not self.dataset:
            # Fallback questions if dataset not available
            fallback_questions = {
                1: {"question": "Explain how to use VLOOKUP function in Excel with an example.", "type": "Beginner"},
                2: {"question": "How would you create a pivot table to analyze sales data by region and product?", "type": "Intermediate"},
                3: {"question": "Describe how to use INDEX-MATCH functions and when they're preferred over VLOOKUP.", "type": "Advanced"}
            }
            return fallback_questions.get(round_num, fallback_questions[1])
            
        round_config = self.round_configs.get(round_num, {})
        target_types = round_config.get("question_types", ["Beginner"])
        
        matching_questions = [
            q for q in self.dataset 
            if q.get("type", "").lower() in [t.lower() for t in target_types]
        ]
        
        if matching_questions:
            return random.choice(matching_questions)
        return random.choice(self.dataset) if self.dataset else fallback_questions[1]
    
    def _score_with_ml_model(self, question, answer):
        """Score answer with ML model - same as original"""
        if not self.scoring_model or not self.vectorizer:
            return self._fallback_scoring(question, answer)
        
        try:
            answer_features = self.vectorizer.transform([answer])
            scores = self.scoring_model.predict(answer_features)[0]
            
            return {
                "correctness": round(max(0, min(100, scores[0] * 10)), 1),
                "clarity": round(max(0, min(100, scores[1] * 10)), 1), 
                "terminology": round(max(0, min(100, scores[2] * 10)), 1),
                "efficiency": round(max(0, min(100, scores[3] * 10)), 1),
                "overall": round(max(0, min(100, scores.mean() * 10)), 1)
            }
        except Exception as e:
            print(f"ML scoring error: {e}")
            return self._fallback_scoring(question, answer)
    
    def _fallback_scoring(self, question, answer):
        """Enhanced fallback scoring"""
        if not answer or len(answer.strip()) < 5:
            return {"correctness": 20, "clarity": 30, "terminology": 25, "efficiency": 20, "overall": 24}
        
        # More sophisticated fallback scoring based on answer length and keywords
        answer_lower = answer.lower()
        word_count = len(answer.split())
        
        # Base score calculation
        base_score = min(85, 40 + word_count * 1.5)
        
        # Bonus for Excel-specific keywords
        excel_keywords = ['vlookup', 'pivot', 'formula', 'function', 'cell', 'range', 'sheet', 'chart', 'data', 'filter']
        keyword_bonus = sum(2 for keyword in excel_keywords if keyword in answer_lower)
        
        final_base = min(90, base_score + keyword_bonus)
        
        return {
            "correctness": round(final_base + random.uniform(-8, 12), 1),
            "clarity": round(final_base + random.uniform(-10, 8), 1),
            "terminology": round(final_base + random.uniform(-5, 15), 1), 
            "efficiency": round(final_base + random.uniform(-12, 6), 1),
            "overall": round(final_base + random.uniform(-5, 10), 1)
        }

# Initialize the RAG system
rag_system = RAGExcelInterviewAPI()

# All routes remain the same...
@app.route('/')
def home():
    """Serve the single-screen interview interface"""
    return render_template('index.html')

@app.route('/api/start_interview', methods=['POST'])
def start_interview():
    """Start new interview session - using original style prompts"""
    try:
        session_id = str(uuid.uuid4())
        
        # Use same prompt style as original
        intro_prompt = """You are an AI Excel Interview Specialist. Introduce yourself to the candidate.

Explain that you'll conduct a 3-round Excel skills assessment:
- Round 1: Foundation (3 questions)
- Round 2: Intermediate (3 questions)  
- Round 3: Advanced (2 questions)

Tell them you'll score on 4 criteria (Correctness, Clarity, Terminology, Efficiency) and give immediate feedback.

Explain the process professionally but warmly. Ask if they're ready to begin.

Keep it conversational and under 200 words."""
        
        introduction = rag_system.call_llm(intro_prompt, max_tokens=250)
        
        # Initialize session data
        rag_system.active_sessions[session_id] = {
            "session_id": session_id,
            "start_time": datetime.datetime.now(),
            "current_round": 1,
            "current_question_num": 1,
            "questions_asked": [],
            "responses": [],
            "scores": [],
            "round_scores": {}
        }
        
        session['interview_session_id'] = session_id
        
        return jsonify({
            "session_id": session_id,
            "introduction": introduction,
            "status": "started"
        })
        
    except Exception as e:
        print(f"❌ Error starting interview: {str(e)}")
        return jsonify({"error": f"Failed to start interview: {str(e)}"}), 500

@app.route('/api/get_question', methods=['GET'])
def get_question():
    """Get next question for current round - using original style prompts"""
    session_id = session.get('interview_session_id')
    if not session_id or session_id not in rag_system.active_sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    session_data = rag_system.active_sessions[session_id]
    current_round = session_data["current_round"]
    current_q_num = session_data["current_question_num"]
    
    # Check if round is complete
    round_config = rag_system.round_configs.get(current_round, {})
    max_questions = round_config.get("num_questions", 3)
    
    if current_q_num > max_questions:
        # Move to next round or end interview
        if current_round < 3:
            session_data["current_round"] += 1
            session_data["current_question_num"] = 1
            current_round = session_data["current_round"]
            current_q_num = 1
        else:
            return jsonify({"interview_complete": True})
    
    # Retrieve question
    question_data = rag_system._retrieve_question_for_round(current_round)
    if not question_data:
        return jsonify({"error": "No questions available"}), 500
    
    # Use original style prompt
    round_config = rag_system.round_configs[current_round]
    question_prompt = f"""You're starting Round {current_round}: {round_config['name']}.

This round: {round_config['description']}

{current_q_num} of {max_questions} questions.

Give a brief, encouraging introduction to this round. Keep it under 50 words."""
    
    formatted_question = rag_system.call_llm(question_prompt, max_tokens=80)
    
    # Store question in session
    session_data["questions_asked"].append(question_data)
    
    return jsonify({
        "round": current_round,
        "round_name": round_config['name'],
        "question_number": current_q_num,
        "total_questions": max_questions,
        "question": formatted_question + f"\n\n{question_data['question']}",
        "question_id": len(session_data["questions_asked"]) - 1
    })

@app.route('/api/submit_answer', methods=['POST'])
def submit_answer():
    """Submit answer and get scores/feedback - using original style prompts"""
    session_id = session.get('interview_session_id')
    if not session_id or session_id not in rag_system.active_sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    data = request.get_json()
    answer = data.get('answer', '').strip()
    question_id = data.get('question_id', -1)
    
    session_data = rag_system.active_sessions[session_id]
    
    if question_id >= len(session_data["questions_asked"]):
        return jsonify({"error": "Invalid question ID"}), 400
    
    question_data = session_data["questions_asked"][question_id]
    
    # Score the answer
    scores = rag_system._score_with_ml_model(question_data['question'], answer)
    
    # Use original style feedback prompt
    feedback_prompt = f"""You asked this Excel question: "{question_data['question']}"

They answered: "{answer}"

AI Scores: Correctness {scores['correctness']}/100, Clarity {scores['clarity']}/100, Terminology {scores['terminology']}/100, Efficiency {scores['efficiency']}/100

Give brief, direct feedback as their interviewer:
1. Quick positive comment (if any)
2. Key correction/tip (if needed)  
3. One practical Excel tip

Keep it under 60 words. Be encouraging and conversational."""
    
    feedback = rag_system.call_llm(feedback_prompt, max_tokens=100)
    
    # Store response data
    session_data["responses"].append({
        "question": question_data['question'],
        "answer": answer,
        "scores": scores,
        "feedback": feedback,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    session_data["scores"].append(scores)
    session_data["current_question_num"] += 1
    
    return jsonify({
        "scores": scores,
        "feedback": feedback,
        "question_complete": True
    })

@app.route('/api/interview_summary', methods=['GET'])
def interview_summary():
    """Get final interview summary with results storage"""
    session_id = session.get('interview_session_id')
    if not session_id or session_id not in rag_system.active_sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    session_data = rag_system.active_sessions[session_id]
    
    # Calculate overall statistics
    all_scores = session_data["scores"]
    if not all_scores:
        return jsonify({"error": "No scores available"}), 400
    
    # Calculate averages
    avg_scores = {
        "correctness": sum(s["correctness"] for s in all_scores) / len(all_scores),
        "clarity": sum(s["clarity"] for s in all_scores) / len(all_scores),
        "terminology": sum(s["terminology"] for s in all_scores) / len(all_scores),
        "efficiency": sum(s["efficiency"] for s in all_scores) / len(all_scores),
        "overall": sum(s["overall"] for s in all_scores) / len(all_scores)
    }
    
    # Use original style summary prompt
    overall_avg = sum(avg_scores.values()) / len(avg_scores)
    final_prompt = f"""Interview complete!

Final Scores: Correctness {avg_scores['correctness']:.1f}, Clarity {avg_scores['clarity']:.1f}, Terminology {avg_scores['terminology']:.1f}, Efficiency {avg_scores['efficiency']:.1f}

Overall Average: {overall_avg:.1f}/100

As their interviewer, give them:
1. Overall performance level (Excellent/Good/Fair/Developing)
2. Top 2 strengths  
3. Top 2 improvement areas
4. Brief next steps

Keep it encouraging and under 150 words."""
    
    final_summary = rag_system.call_llm(final_prompt, max_tokens=200)
    
    # Save complete interview results to file
    results_filepath = rag_system.save_interview_results(session_data, final_summary, avg_scores)
    
    return jsonify({
        "summary": final_summary,
        "avg_scores": avg_scores,
        "total_questions": len(all_scores),
        "detailed_responses": session_data["responses"],
        "results_saved": results_filepath is not None,
        "results_filepath": results_filepath
    })

@app.route('/api/get_interview_history', methods=['GET'])
def get_interview_history():
    """Get list of all completed interviews"""
    try:
        if not os.path.exists(rag_system.results_directory):
            return jsonify({"interviews": []})
        
        interviews = []
        for filename in os.listdir(rag_system.results_directory):
            if filename.startswith("interview_") and filename.endswith(".json"):
                filepath = os.path.join(rag_system.results_directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        interviews.append({
                            "session_id": data["session_info"]["session_id"][:8],
                            "date": data["session_info"]["start_time"][:10],
                            "duration": data["session_info"]["total_duration_minutes"],
                            "overall_score": data["summary"]["average_scores"]["overall"],
                            "performance": data["summary"]["overall_performance"],
                            "filename": filename
                        })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
        
        # Sort by date (most recent first)
        interviews.sort(key=lambda x: x["date"], reverse=True)
        
        return jsonify({"interviews": interviews})
        
    except Exception as e:
        print(f"Error getting interview history: {e}")
        return jsonify({"error": "Failed to get interview history"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)