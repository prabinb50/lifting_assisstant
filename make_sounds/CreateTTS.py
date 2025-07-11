from gtts import gTTS

text = "down"
lang = "en"

tts = gTTS(text=text, lang=lang)

tts.save("F:\powerlifter\AI_Exercise_Pose_Feedback\resources\sounds")
