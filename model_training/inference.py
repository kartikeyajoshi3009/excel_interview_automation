import joblib

# Load saved model and vectorizer
reg = joblib.load("ridge_multioutput_regressor_updated_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer_updated_model.joblib")

def score_answer(question, answer, type_):
    # Combine input text fields
    text = question + " " + answer + " " + type_
    # Vectorize the text
    text_vec = vectorizer.transform([text])
    # Predict scores (correctness, clarity, terminology, efficiency)
    predicted_scores = reg.predict(text_vec)[0]
    # Optionally round scores
    rounded_scores = predicted_scores.round(2)
    # Return as dict
    return {
        "correctness": rounded_scores[0],
        "clarity": rounded_scores[1],
        "terminology": rounded_scores[2],
        "efficiency": rounded_scores[3]
    }

# Example usage
question = "How do you save a file in Excel?"
answer = "You can use Ctrl+S to save your work."
type_ = "Advanced"

scores = score_answer(question, answer, type_)
print("Predicted scores:", scores)