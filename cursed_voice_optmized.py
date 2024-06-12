import sounddevice as sd
import numpy as np
import deepspeech
import os
from pydub import AudioSegment
import pygame
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

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

# Load the CUDA-enabled DeepSpeech model
model_path = 'deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_path)

target_sentence = "test"
target_words = target_sentence.split()
detected_words = []
word_timestamps = []

print("Recording... Press 'q' to stop.")

def record_audio(audio_queue, stop_event):
    with sd.InputStream(channels=CHANNELS, samplerate=RATE, blocksize=CHUNK, dtype='int16'):
        while not stop_event.is_set():
            # Record audio using sounddevice
            audio_data = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=CHANNELS, dtype='int16')
            sd.wait()
            
            audio_queue.put(audio_data.copy())

def process_audio(audio_queue, stop_event):
    while not stop_event.is_set() and len(detected_words) < len(target_words):
        try:
            audio_data = audio_queue.get(timeout=1)
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
        except Empty:
            pass
    
    stop_event.set()

stop_event = threading.Event()
audio_queue = Queue()

# Start audio recording in a separate thread
recording_thread = threading.Thread(target=record_audio, args=(audio_queue, stop_event))
recording_thread.start()

# Process audio using multiple threads
num_processing_threads = min(os.cpu_count(), 4)  # Limit the number of threads to 4
with ThreadPoolExecutor(max_workers=num_processing_threads) as executor:
    futures = []
    for _ in range(num_processing_threads):
        future = executor.submit(process_audio, audio_queue, stop_event)
        futures.append(future)
    
    # Wait for all processing tasks to complete or stop event to be set
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
    
    # Perform speech recognition on the audio segment to get the timing information
    result = model.sttWithMetadata(audio_data)
    timing_info = result.transcripts[0].tokens if result.transcripts else []
    
    # Find the start and end timestamps of the target word within the audio segment
    word_start_time = None
    word_end_time = None
    for token in timing_info:
        if token.text.lower() == word:
            word_start_time = token.start_time
            word_end_time = token.start_time + token.duration
            break
    
    if word_start_time is not None and word_end_time is not None:
        # Extract the word audio segment based on the timestamps
        word_audio_segment = audio_segment[word_start_time * 1000:word_end_time * 1000]
        
        # Append the word audio segment to the output audio
        output_audio += word_audio_segment
    else:
        print(f"Word '{word}' not found in the audio segment.")

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

# Clean up resources
model.freeModel()
