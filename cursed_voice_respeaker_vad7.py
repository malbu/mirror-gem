import numpy as np
import deepspeech
import os
from pydub import AudioSegment
import threading
from queue import Queue, Empty
import usb.core
import usb.util
from tuning import Tuning
import pyaudio
import time
import sounddevice as sd

# Set the audio recording parameters
CHUNK = 512
CHANNELS = 1  # Record from a single channel
RATE = 16000
RESPEAKER_INDEX = 11  
WAVE_OUTPUT_FOLDER = "audio_clips"
OUTPUT_AUDIO_FILE = os.path.join(WAVE_OUTPUT_FOLDER, "output_audio.wav")

# Create the audio clips folder if it doesn't exist
if not os.path.exists(WAVE_OUTPUT_FOLDER):
    os.makedirs(WAVE_OUTPUT_FOLDER)

# Load the CUDA-enabled DeepSpeech model
model_path = 'deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_path)

target_sentence = "help me"
target_words = target_sentence.split()
detected_words = []
word_timestamps = []

print("Waiting for voice activity...")

def record_audio(audio_queue, stop_event, tuning):
    chunk_counter = 0
    p = pyaudio.PyAudio()

    buffer = []
    voice_detected = False

    def audio_callback(in_data, frame_count, time_info, status):
        nonlocal buffer, voice_detected
        audio_data = np.frombuffer(in_data, dtype=np.int16)[0::6]  # Extract channel 0
        buffer.extend(audio_data)
        if tuning.is_voice():
            voice_detected = True
        return (in_data, pyaudio.paContinue)

    stream = p.open(
        rate=RATE,
        format=p.get_format_from_width(2),
        channels=6,
        input=True,
        input_device_index=RESPEAKER_INDEX,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )

    stream.start_stream()

    MIN_DURATION = 5  # Minimum duration in seconds
    min_samples = MIN_DURATION * RATE

    while not stop_event.is_set():
        if len(buffer) >= min_samples:
            if voice_detected:
                audio_data = np.array(buffer[:min_samples], dtype=np.int16)
                
                chunk_audio_file = os.path.join(WAVE_OUTPUT_FOLDER, f"chunk_{chunk_counter}.wav")
                chunk_audio = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=RATE,
                    sample_width=audio_data.dtype.itemsize,
                    channels=1
                )
                chunk_audio.export(chunk_audio_file, format="wav")
                print(f"Recorded audio chunk with voice activity saved as {chunk_audio_file}")
                
                audio_queue.put(audio_data)
                chunk_counter += 1

            buffer = buffer[min_samples:]
            voice_detected = False

        time.sleep(0.1)  # Small delay to prevent busy-waiting

    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio(audio_queue, stop_event):
    processed_counter = 0
    while not stop_event.is_set() and len(detected_words) < len(target_words):
        try:
            audio_data = audio_queue.get(timeout=1)
            
            # Save the processed audio data as a separate WAV file
            processed_audio_file = os.path.join(WAVE_OUTPUT_FOLDER, f"processed_{processed_counter}.wav")
            processed_audio = AudioSegment(
                audio_data.tobytes(),
                frame_rate=RATE,
                sample_width=audio_data.dtype.itemsize,
                channels=1
            )
            processed_audio.export(processed_audio_file, format="wav")
            print(f"Processed audio saved as {processed_audio_file}")
            
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
            processed_counter += 1
        except Empty:
            pass
    
    stop_event.set()

def start_recording(tuning):
    stop_event = threading.Event()
    audio_queue = Queue()

    # Start audio recording in a separate thread
    recording_thread = threading.Thread(target=record_audio, args=(audio_queue, stop_event, tuning))
    recording_thread.start()

    # Process audio using multiple threads
    num_processing_threads = min(os.cpu_count(), 4)  # Limit the number of threads to 4
    processing_threads = []
    for _ in range(num_processing_threads):
        thread = threading.Thread(target=process_audio, args=(audio_queue, stop_event))
        thread.start()
        processing_threads.append(thread)
    
    # Wait for all processing tasks to complete or stop event to be set
    for thread in processing_threads:
        thread.join()

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
            channels=1
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
                word_start_time = timing_info[i].start_time-.2
                word_end_time = timing_info[i+len(word)-1].start_time+.2
                break
        
        if word_start_time is not None and word_end_time is not None:
            # Extract the word audio segment based on the timestamps
            word_audio_segment = audio_segment[word_start_time * 1000:word_end_time * 1000]
            
            # Append the word audio segment to the output audio
            output_audio += word_audio_segment
            
            print(f"Extracted audio segment for word '{word}' with start time {word_start_time} and end time {word_end_time}")

        # Export the output audio as a WAV file
            output_audio.export(OUTPUT_AUDIO_FILE, format="wav")
            print(f"Output audio saved as {OUTPUT_AUDIO_FILE}")

        else:
            print(f"Word '{word}' not found in the audio segment. Skipping.")
        
        print()  # Add a blank line for readability

    # Play the output audio file
    output_audio = AudioSegment.from_wav(OUTPUT_AUDIO_FILE)
    sd.play(output_audio.get_array_of_samples(), output_audio.frame_rate)
    sd.wait()

# Main entry point
if __name__ == "__main__":
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if dev:
        tuning = Tuning(dev)
        tuning.write('GAMMAVAD_SR', 15)  # Adjust VAD sensitivity
        try:
            start_recording(tuning)
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            tuning.close()
    else:
        print("ReSpeaker not found.")
