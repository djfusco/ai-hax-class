import pyttsx3

# Initialize the engine
engine = pyttsx3.init()

# Customize voice (optional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Change index for different voices
engine.setProperty('rate', 150)  # Speed (words per minute)

# Your text
text = "Welcome to my podcast! Today, we’ll talk about creating audio with Python."

# Save to file
engine.save_to_file(text, "podcast.mp3")
engine.runAndWait()