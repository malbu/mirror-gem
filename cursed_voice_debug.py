import sounddevice as sd
import numpy as np
import deepspeech
import datetime
import os
from pydub import AudioSegment
import pygame
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

# Set the audio recording parameters
CHUNK = 512
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FOLDER = "audio_clips"
OUTPUT_AUDIO_FILE = os.path.join(WAVE_OUTPUT_FOLDER, "output_audio.wav")

# Create the audio clips folder if it doesn't exist
if not os.path.exists(WAVE_OUTPUT_FOLDER):
    os.makedirs(WAVE_OUTPUT_FOLDER)

# Load the CUDA-enabled DeepSpeech model and scorer
model_path = 'deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_path)

target_sentence = "test"
target_words = target_sentence.split()
detected_words = []
word_timestamps = []

print("Recording... Press 'q' to stop.")

def record_audio(audio_queue):
    chunk_counter = 0
    while True:
        # Record audio using sounddevice
        audio_data = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=CHANNELS, dtype='int16')
        sd.wait()
        
        # Save the recorded audio chunk as a separate WAV file
        chunk_audio_file = os.path.join(WAVE_OUTPUT_FOLDER, f"chunk_{chunk_counter}.wav")
        chunk_audio = AudioSegment(
            audio_data.tobytes(),
            frame_rate=RATE,
            sample_width=audio_data.dtype.itemsize,
            channels=CHANNELS
        )
        chunk_audio.export(chunk_audio_file, format="wav")
        print(f"Recorded audio chunk saved as {chunk_audio_file}")
        
        audio_queue.put(audio_data.copy())
        chunk_counter += 1

def process_audio(audio_queue):
    while len(detected_words) < len(target_words):
        audio_data = audio_queue.get()
        # Reshape the audio data to have a single dimension
        audio_data = audio_data.reshape(-1)
        
        # Perform speech recognition using DeepSpeech with GPU acceleration
        result = model.stt(audio_data)
        text = result
        
        # Check if any target words are detected in the recognized text
        for word in text.split():
            if word.lower() in target_words and word.lower() not in detected_words:
                detected_words.append(word.lower())
                print(f"Detected word: {word}")
                print()
                word_timestamps.append((word.lower(), audio_data))
        
        print(f"Recognized speech: {text}")
        print()
    
    audio_queue.task_done()

audio_queue = Queue()

# Start audio recording in a separate thread
recording_thread = threading.Thread(target=record_audio, args=(audio_queue,))
recording_thread.daemon = True
recording_thread.start()

# Process audio using multiple threads
with ThreadPoolExecutor() as executor:
    futures = []
    for _ in range(os.cpu_count()):
        future = executor.submit(process_audio, audio_queue)
        futures.append(future)
    
    # Wait for all processing tasks to complete
    for future in futures:
        future.result()

print("Target sentence detected.")

# Create an empty audio segment to store the output audio
output_audio = AudioSegment.empty()

# Iterate over the word timestamps and extract the corresponding audio segments
for i, (word, audio_data) in enumerate(word_timestamps):
    # Convert audio data to AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=RATE,
        sample_width=audio_data.dtype.itemsize,
        channels=CHANNELS
    )
    # Save the word audio segment as a separate WAV file
    word_audio_file = os.path.join(WAVE_OUTPUT_FOLDER, f"word_{i+1}.wav")
    audio_segment.export(word_audio_file, format="wav")
    print(f"Word '{word}' audio saved as {word_audio_file}")
    
    # Append the word audio to the output audio
    output_audio += audio_segment

# Export the output audio as a WAV file
output_audio.export(OUTPUT_AUDIO_FILE, format="wav")
print(f"Output audio saved as {OUTPUT_AUDIO_FILE}")

# Play the output audio file
pygame.mixer.init()
pygame.mixer.music.load(OUTPUT_AUDIO_FILE)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

pygame.mixer.quit()
