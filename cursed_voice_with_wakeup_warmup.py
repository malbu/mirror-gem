import numpy as np
import deepspeech
import os
import pyaudio
import wave
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import webrtcvad
import pygame
from functools import wraps
from time import time as now
import logging

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

# Load the DeepSpeech model with TensorRT support
model_path = 'deepspeech-0.8.2-models.pbmm'
model = deepspeech.Model(model_path, use_trt=True)

target_sentence = "help me"
target_words = target_sentence.split()
detected_words = []
word_timestamps = []

print("Initializing voice activity detection...")

# Initialize WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(1)  # 0 is least aggressive, 3 is most aggressive

# Initialize PyAudio
audio = pyaudio.PyAudio()

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = now()
        result = func(*args, **kwargs)
        stop = now()
        logging.debug('{} took {:.3f}s'.format(func.__name__, stop-start))
        return result
    return wrapper

@timed
def transcribe(model, audio, detected_words, word_timestamps):
    audio_data = np.frombuffer(audio, dtype=np.int16)
    result = model.stt(audio_data)
    text = result

    # Check if any target words are detected in the recognized text
    for word in text.split():
        if word.lower() in target_words and word.lower() not in detected_words:
            detected_words.append(word.lower())
            logging.info(f"Detected word: {word}")
            word_timestamps.append((word.lower(), audio_data))

@timed
def stream_callback(indata, frames, time, status, audio_queue):
    if status:
        logging.warning(f"Stream callback status: {status}")
    audio_queue.put(indata)

@timed
def process_audio(audio_queue, model, detected_words, word_timestamps):
    try:
        audio = audio_queue.get(timeout=1)
        transcribe(model=model, audio=audio, detected_words=detected_words, word_timestamps=word_timestamps)
    except Empty:
        pass

def vad_listener():
    stream = audio.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("Listening for speech... Press Ctrl+C to stop.")
    try:
        while True:
            indata = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(indata, RATE)
            if is_speech:
                print("Speech detected!")
                stop_event.set()  # Stop the VAD detection
                start_recording()
                break
    except KeyboardInterrupt:
        print("Stopping speech detection...")
    finally:
        stream.stop_stream()
        stream.close()

def record_audio(audio_queue, stop_event):
    chunk_counter = 0
    stream = audio.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    try:
        while not stop_event.is_set():
            # Record audio using PyAudio
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Save the recorded audio chunk as a separate WAV file
            chunk_audio_file = os.path.join(WAVE_OUTPUT_FOLDER, f"chunk_{chunk_counter}.wav")
            with wave.open(chunk_audio_file, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(RATE)
                wf.writeframes(audio_data)
            print(f"Recorded audio chunk saved as {chunk_audio_file}")
            
            audio_queue.put(audio_data)
            chunk_counter += 1
    finally:
        stream.stop_stream()
        stream.close()

def start_recording():
    stop_event.clear()
    audio_queue = Queue()
    # Start audio recording in a separate thread
    recording_thread = threading.Thread(target=record_audio, args=(audio_queue, stop_event))
    recording_thread.start()

    # Process audio using multiple threads
    num_processing_threads = min(os.cpu_count(), 4)  # Limit the number of threads to 4
    with ThreadPoolExecutor(max_workers=num_processing_threads) as executor:
        futures = [executor.submit(process_audio, audio_queue, model, detected_words, word_timestamps) for _ in range(num_processing_threads)]
        
        # Wait for all processing tasks to complete or stop event to be set
        for future in futures:
            future.result()

    print("Target sentence detected.")
    process_detected_words()

def process_detected_words():
    # Create an empty audio segment to store the output audio
    output_audio = b""

    # Iterate over the word timestamps and extract the corresponding audio segments
    for i, (word, audio_data) in enumerate(word_timestamps):
        # Perform speech recognition on the audio segment to get the timing information
        result = model.sttWithMetadata(audio_data)
        timing_info = result.transcripts[0].tokens if result.transcripts else []

        # Print the metadata for each token
        print(f"Metadata for word '{word}':")
        for token in timing_info:
            print(f"Word: {token.text}, Start Time: {token.start_time}")

        # Find the start and end timestamps of the target word within the audio segment
        word_start_time = None
        word_end_time = None

        for j in range(len(timing_info) - len(word.split()) + 1):
            if ''.join(token.text.lower() for token in timing_info[j:j+len(word.split())]) == word:
                word_start_time = timing_info[j].start_time - 0.2
                word_end_time = timing_info[j+len(word.split())-1].start_time + 0.2
                break

        if word_start_time is not None and word_end_time is not None:
            # Extract the word audio segment based on the timestamps
            start_frame = int(word_start_time * RATE)
            end_frame = int(word_end_time * RATE)
            word_audio_segment = audio_data[start_frame:end_frame]
            
            # Append the word audio segment to the output audio
            output_audio += word_audio_segment.tobytes()
            
            print(f"Extracted audio segment for word '{word}' with start time {word_start_time} and end time {word_end_time}")

        else:
            print(f"Word '{word}' not found in the audio segment. Skipping.")

    # Save the output audio as a WAV file
    with wave.open(OUTPUT_AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(output_audio)
    print(f"Output audio saved as {OUTPUT_AUDIO_FILE}")

    # Play the output audio file
    pygame.mixer.init()
    pygame.mixer.music.load(OUTPUT_AUDIO_FILE)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

# Main entry point
if __name__ == "__main__":
    stop_event = threading.Event()
    logging.basicConfig(level=logging.INFO)
    try:
        print("Listening for speech... Press Ctrl+C to stop.")
        vad_listener()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        audio.terminate()

