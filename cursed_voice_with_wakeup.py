import sounddevice as sd
import numpy as np
import deepspeech
import os
from pydub import AudioSegment
import pygame
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import webrtcvad

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

def vad_callback(indata, frames, time, status):
    if status:
        print(status)
    is_speech = vad.is_speech(indata.tobytes(), RATE)
    if is_speech:
        print("Speech detected!")
        stop_event.set()  # Stop the VAD detection
        start_recording()

# Initialize audio stream for VAD
def vad_listener():
    with sd.InputStream(channels=1, samplerate=RATE, blocksize=CHUNK, callback=vad_callback):
        print("Listening for speech... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopping speech detection...")

def record_audio(audio_queue, stop_event):
    chunk_counter = 0
    with sd.InputStream(channels=CHANNELS, samplerate=RATE, blocksize=CHUNK, dtype='int16') as stream:
        while not stop_event.is_set():
            # Record audio using sounddevice
            audio_data = stream.read(CHUNK)[0]
            
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

def process_audio(audio_queue, stop_event):
    processed_counter = 0
    while not stop_event.is_set() and len(detected_words) < len(target_words):
        try:
            audio_data = audio_queue.get(timeout=1)
            # Reshape the audio data to have a single dimension
            audio_data = audio_data.reshape(-1)
            
            # Save the processed audio data as a separate WAV file
            processed_audio_file = os.path.join(WAVE_OUTPUT_FOLDER, f"processed_{processed_counter}.wav")
            processed_audio = AudioSegment(
                audio_data.tobytes(),
                frame_rate=RATE,
                sample_width=audio_data.dtype.itemsize,
                channels=CHANNELS
            )
            processed_audio.export(processed_audio_file, format="wav")
            print(f"Processed audio saved as {processed_audio_file}")
            
            # Perform speech recognition using DeepSpeech with TensorRT acceleration
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
            processed_counter += 1
        except Empty:
            pass
        
    stop_event.set()

def start_recording():
    stop_event.clear()
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
        
        # Print the metadata for each token
        print(f"Metadata for word '{word}':")
        for token in timing_info:
            print(f"Word: {token.text}, Start Time: {token.start_time}")
        
        # Find the start and end timestamps of the target word within the audio segment
        word_start_time = None
        word_end_time = None
        
        for i in range(len(timing_info) - len(word) + 1):
            if ''.join(token.text.lower() for token in timing_info[i:i+len(word)]) == word:
                word_start_time = timing_info[i].start_time - 0.2
                word_end_time = timing_info[i+len(word)-1].start_time + 0.2
                break
        
        if word_start_time is not None and word_end_time is not None:
            # Extract the word audio segment based on the timestamps
            word_audio_segment = audio_segment[word_start_time * 1000:word_end_time * 1000]
            
            # Append the word audio segment to the output audio
            output_audio += word_audio_segment
            
            print(f"Extracted audio segment for word '{word}' with start time {word_start_time} and end time {word_end_time}")

        else:
            print(f"Word '{word}' not found in the audio segment. Skipping.")
        
        print()  # Add a blank line for readability

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

# Main entry point
if __name__ == "__main__":
    stop_event = threading.Event()
    try:
        print("Listening for speech... Press Ctrl+C to stop.")
        vad_listener()
    except KeyboardInterrupt:
        print("Exiting...")