from gtts import gTTS
import os

# Dictionary of all feedback messages and their file paths
audio_files = {
    # Excessive arch feedback
    "./resources/sounds/excessive_arch_1.mp3": 
        "Don't arch your back too much. Try to focus on expanding your chest.",
    "./resources/sounds/excessive_arch_2.mp3": 
        "Lift your pelvis slightly more and tighten your abs to keep your back flat.",
    
    # Arms spread feedback
    "./resources/sounds/arms_spread_1.mp3": 
        "You're gripping the bar too wide. Narrow your grip a bit.",
    "./resources/sounds/arms_spread_2.mp3": 
        "When gripping the bar, it's better to hold it just slightly wider than shoulder width.",
    
    # Spine neutral feedback
    "./resources/sounds/spine_neutral_feedback_1.mp3": 
        "Try not to excessively bend your spine.",
    "./resources/sounds/spine_neutral_feedback_2.mp3": 
        "Lift your chest and pull your shoulders back.",
    
    # Caved in knees feedback
    "./resources/sounds/caved_in_knees_feedback_1.mp3": 
        "Be careful not to let your knees cave inward.",
    "./resources/sounds/caved_in_knees_feedback_2.mp3": 
        "Push your hips back to maintain alignment between your knees and toes.",
    
    # Other feedback
    "./resources/sounds/feet_spread.mp3": 
        "Narrow your stance to keep your feet about shoulder-width apart.",
    "./resources/sounds/arms_narrow.mp3": 
        "It's better to grip the bar slightly wider than shoulder width.",
    "./resources/sounds/correct.mp3": 
        "You are performing the exercise with correct posture."
}

# Generate audio files
for file_path, text in audio_files.items():
    print(f"Generating: {file_path}")
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(file_path)
    print(f"Saved: {file_path}")

print("All audio files have been generated successfully!")