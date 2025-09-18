// Single-Screen Excel Interview System - Updated JavaScript
class ExcelInterviewApp {
    constructor() {
        this.currentSession = null;
        this.currentQuestionId = null;
        this.isLoading = false;
        this.state = 'welcome'; // welcome, intro, question, feedback, summary
        
        this.init();
    }
    
    init() {
        this.bindEvents();
    }
    
    bindEvents() {
        // Dynamic action button
        document.getElementById('action-btn').addEventListener('click', () => {
            this.handleActionButton();
        });
        
        // Restart button
        document.getElementById('restart-btn').addEventListener('click', () => {
            this.restartInterview();
        });
        
        // Character counter for textarea
        const textarea = document.getElementById('user-answer');
        textarea.addEventListener('input', () => {
            this.updateCharacterCount();
            this.updateActionButton();
        });
        
        // Error retry
        document.getElementById('error-retry-btn').addEventListener('click', () => {
            this.hideError();
        });
        
        // Enter key handling
        textarea.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                if (this.state === 'question') {
                    this.handleActionButton();
                }
            }
        });
    }
    
    handleActionButton() {
        switch (this.state) {
            case 'welcome':
                this.startInterview();
                break;
            case 'intro':
                this.loadNextQuestion();
                break;
            case 'question':
                this.submitAnswer();
                break;
            case 'feedback':
                this.loadNextQuestion();
                break;
        }
    }
    
    updateActionButton() {
        const btn = document.getElementById('action-btn');
        const textarea = document.getElementById('user-answer');
        
        switch (this.state) {
            case 'welcome':
                btn.textContent = 'Start Interview';
                btn.disabled = false;
                break;
            case 'intro':
                btn.textContent = 'Continue to Questions';
                btn.disabled = false;
                break;
            case 'question':
                if (textarea.value.trim().length >= 10) {
                    btn.textContent = 'Submit Answer';
                    btn.disabled = false;
                } else {
                    btn.textContent = 'Enter at least 10 characters';
                    btn.disabled = true;
                }
                break;
            case 'feedback':
                btn.textContent = 'Next Question';
                btn.disabled = false;
                break;
        }
    }
    
    setState(newState) {
        this.state = newState;
        this.updateActionButton();
        this.updateVisibility();
    }
    
    updateVisibility() {
        // Hide all sections
        document.getElementById('answer-section').classList.add('hidden');
        document.getElementById('scores-section').classList.add('hidden');
        document.getElementById('final-scores-section').classList.add('hidden');
        document.getElementById('restart-btn').classList.add('hidden');
        
        // Show relevant sections based on state
        switch (this.state) {
            case 'question':
                document.getElementById('answer-section').classList.remove('hidden');
                break;
            case 'feedback':
                document.getElementById('scores-section').classList.remove('hidden');
                break;
            case 'summary':
                document.getElementById('final-scores-section').classList.remove('hidden');
                document.getElementById('restart-btn').classList.remove('hidden');
                document.getElementById('action-btn').classList.add('hidden');
                break;
        }
    }
    
    showLoading(show = true) {
        const loading = document.getElementById('loading');
        if (show) {
            loading.classList.remove('hidden');
            this.isLoading = true;
        } else {
            loading.classList.add('hidden');
            this.isLoading = false;
        }
    }
    
    showError(message) {
        document.getElementById('error-text').textContent = message;
        document.getElementById('error-message').classList.remove('hidden');
    }
    
    hideError() {
        document.getElementById('error-message').classList.add('hidden');
    }
    
    async startInterview() {
        if (this.isLoading) return;
        
        try {
            this.showLoading(true);
            
            const response = await fetch('/api/start_interview', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            
            this.currentSession = data.session_id;
            
            // Update content area with introduction
            document.getElementById('content-area').textContent = data.introduction;
            this.setState('intro');
            
        } catch (error) {
            console.error('Error starting interview:', error);
            this.showError('Failed to start interview. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    async loadNextQuestion() {
        if (this.isLoading) return;
        
        try {
            this.showLoading(true);
            
            const response = await fetch('/api/get_question');
            const data = await response.json();
            
            if (data.error) throw new Error(data.error);
            
            if (data.interview_complete) {
                this.showInterviewSummary();
                return;
            }
            
            // Update session info
            this.updateSessionInfo(data.round, data.round_name, data.question_number, data.total_questions);
            
            // Update content area with question
            document.getElementById('content-area').textContent = data.question;
            this.currentQuestionId = data.question_id;
            
            // Clear previous answer and set state
            document.getElementById('user-answer').value = '';
            this.updateCharacterCount();
            this.setState('question');
            
            // Focus on textarea
            document.getElementById('user-answer').focus();
            
        } catch (error) {
            console.error('Error loading question:', error);
            this.showError('Failed to load question. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    async submitAnswer() {
        if (this.isLoading) return;
        
        const answer = document.getElementById('user-answer').value.trim();
        
        if (answer.length < 10) {
            this.showError('Please provide a more detailed answer (at least 10 characters).');
            return;
        }
        
        try {
            this.showLoading(true);
            
            const response = await fetch('/api/submit_answer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    answer: answer,
                    question_id: this.currentQuestionId
                })
            });
            
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            
            // Update content area with feedback
            document.getElementById('content-area').textContent = data.feedback;
            
            // Display scores
            this.displayScores(data.scores);
            this.setState('feedback');
            
        } catch (error) {
            console.error('Error submitting answer:', error);
            this.showError('Failed to submit answer. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    async showInterviewSummary() {
        try {
            this.showLoading(true);
            
            const response = await fetch('/api/interview_summary');
            const data = await response.json();
            
            if (data.error) throw new Error(data.error);
            
            // Update content area with summary
            document.getElementById('content-area').textContent = data.summary;
            
            // Display final scores
            this.displayFinalScores(data.avg_scores);
            
            // Hide session info
            document.getElementById('session-info').classList.add('hidden');
            this.setState('summary');
            
        } catch (error) {
            console.error('Error loading summary:', error);
            this.showError('Failed to load interview summary.');
        } finally {
            this.showLoading(false);
        }
    }
    
    updateSessionInfo(round, roundName, questionNum, totalQuestions) {
        document.getElementById('round-info').textContent = `${roundName} (Round ${round}/3)`;
        document.getElementById('question-counter').textContent = `Question ${questionNum}/${totalQuestions}`;
        document.getElementById('session-info').classList.remove('hidden');
    }
    
    displayScores(scores) {
        document.getElementById('score-correctness').textContent = scores.correctness.toFixed(1);
        document.getElementById('score-clarity').textContent = scores.clarity.toFixed(1);
        document.getElementById('score-terminology').textContent = scores.terminology.toFixed(1);
        document.getElementById('score-efficiency').textContent = scores.efficiency.toFixed(1);
        document.getElementById('score-overall').textContent = scores.overall.toFixed(1);
    }
    
    displayFinalScores(avgScores) {
        document.getElementById('final-correctness').textContent = avgScores.correctness.toFixed(1);
        document.getElementById('final-clarity').textContent = avgScores.clarity.toFixed(1);
        document.getElementById('final-terminology').textContent = avgScores.terminology.toFixed(1);
        document.getElementById('final-efficiency').textContent = avgScores.efficiency.toFixed(1);
        document.getElementById('final-overall').textContent = avgScores.overall.toFixed(1);
    }
    
    updateCharacterCount() {
        const textarea = document.getElementById('user-answer');
        const charCount = document.getElementById('char-count');
        charCount.textContent = textarea.value.length;
        
        if (textarea.value.length > 1800) {
            charCount.style.color = '#dc3545';
        } else if (textarea.value.length > 1500) {
            charCount.style.color = '#ffc107';
        } else {
            charCount.style.color = '#6c757d';
        }
    }
    
    restartInterview() {
        // Reset state
        this.currentSession = null;
        this.currentQuestionId = null;
        
        // Reset content
        document.getElementById('content-area').innerHTML = `
            <h2>Welcome to Your Excel Skills Assessment</h2>
            <p>Click "Start Interview" to begin your personalized Excel skills evaluation.</p>
        `;
        
        // Clear form
        document.getElementById('user-answer').value = '';
        this.updateCharacterCount();
        
        // Reset scores
        const scoreElements = ['correctness', 'clarity', 'terminology', 'efficiency', 'overall'];
        scoreElements.forEach(score => {
            document.getElementById(`score-${score}`).textContent = '-';
            document.getElementById(`final-${score}`).textContent = '-';
        });
        
        // Hide session info
        document.getElementById('session-info').classList.add('hidden');
        
        // Reset to welcome state
        this.setState('welcome');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new ExcelInterviewApp();
});

// Prevent accidental page refresh during interview
window.addEventListener('beforeunload', (e) => {
    const app = window.app;
    if (app && (app.state === 'question' || app.state === 'feedback')) {
        e.preventDefault();
        e.returnValue = 'Are you sure you want to leave? Your interview progress will be lost.';
        return e.returnValue;
    }
});