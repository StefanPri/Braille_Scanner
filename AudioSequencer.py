# gTTS (Google Text-to-Speech) is a Python library and CLI tool to interface with Google Translate's text-to-speech API
from gtts import gTTS  
# PyDub is a simple and easy-to-use library to manipulate audio with a simple and easy to use interface
from pydub import AudioSegment  
# The OS module in Python provides functions for interacting with the operating system
import os  
# Playsound is a pure Python, cross-platform, single function module with no dependencies for playing sounds
from playsound import playsound  
# Time module provides various time-related functions
import time

testSentence = "This is a test sentence for the audio sequencer."

def audioSequencer(sentence):

    # Split the sentence into words
    words = sentence.split(' ')
    # Initialize an empty list to store the paths of the audio files
    audioFiles = []
    # Count the number of files in the current directory
    audioCounter = len([name for name in os.listdir('.') if os.path.isfile(name)])
    # Define the name of the directory where the audio files will be stored
    audioFolder = "Audio"
    # If the directory does not exist, create it
    if not os.path.exists(audioFolder):
        os.makedirs(audioFolder)

    # Generate audio for each word
    for word in words:
        # Define the path of the audio file for the current word
        audioFile = f"{audioFolder}/{audioCounter}_{word}.mp3"
        # If the audio file does not exist, create it
        if not os.path.isfile(audioFile):
            # Use Google Text-to-Speech to generate the audio
            tts = gTTS(text=word, lang='en')
            # Save the audio file
            tts.save(audioFile)
            # Play the audio before saving
            playsound(audioFile)
        # Add the path of the audio file to the list
        audioFiles.append(audioFile)
        # Increment the counter
        audioCounter += 1

    # Initialize an empty audio segment
    combinedAudio = AudioSegment.empty()
    # Concatenate the audio files
    for audioFile in audioFiles:
        combinedAudio += AudioSegment.from_mp3(audioFile)

    # Save the combined audio
    combinedAudio.export(f"{audioFolder}/{audioCounter}_combined_audio.mp3", format='mp3')

st = time.time()
audioSequencer(testSentence)
en = time.time()
print(en-st)