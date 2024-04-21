
from gtts import gTTS
import os

text = """

Hello, how are you today?
This is a trial of me using gTTS.
It converts an input text into audio and saves the audio file on my computer.

"""

# Language in which you want to convert
language = 'en'

# Specify the full path where you want to save the output file
output_file_path = r"E:\NLP Learning\NLP-Learning\gTTS Experiment\Audio Output.mp3"
# output_file_path = "E:\\NLP Learning\\NLP-Learning\\gTTS Experiment\\Audio Output.mp3"

# Passing the text and language to the engine, slow=False makes the speech faster
speech = gTTS(text=text, lang=language, slow=False)

# # Save the converted audio to a file in the same folder
# speech.save("Audio Output.mp3")

# Save the converted audio to the specified path
speech.save(output_file_path)

# Print confirmation message
print(f"File saved at: {output_file_path}")

# # Play the converted file
# os.system("start Audio Output.mp3")