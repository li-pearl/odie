import google.generativeai as genai
from SensitiveConstants import GEMINI_API_KEY
import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini model
model = genai.GenerativeModel("gemini-pro")  # You can use other Gemini models if available

# Function to generate a one-sentence summary of visual context
def generate_visual_context_summary(speech_text, pose_label, emotion_label, gesture_label):
    prompt = f"""
    Here is the information available from the other person talking to the user:
    - **Speech from the other person in the conversation**: "{speech_text}"
    - **Pose**: {pose_label}
    - **Emotion**: {emotion_label}
    - **Gesture**: {gesture_label}

    You are assisting a blind or neurodivergent user by providing a concise one-sentence summary of the visual context they are missing using the information provided. In addition to the analysis this sentence should provide, it should also include all pieces of information available except for the speech in the sentence.
    The summary should be natural, clear, and informative without unnecessary details. But, it should give context to the user about visual information. For example, if the speech is a greeting and a hand is raised, it means they are waving. 
    """

    # Send prompt to Gemini API
    response = model.generate_content(prompt)

    return response.text  # Get the AI-generated response

# Example data (Replace with real values)
speech_text = "Raya has a ball!"
pose_label = "Open stance"
emotion_label = "Happy"
gesture_label = "Right Hand Raised"

summary = generate_visual_context_summary(speech_text, pose_label, emotion_label, gesture_label)
print(summary)
speak(summary)