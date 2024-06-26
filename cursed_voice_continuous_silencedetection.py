import numpy as np
import deepspeech
import os
from pydub import AudioSegment
import threading
from queue import Queue, Empty, Full
import usb.core
import usb.util
from tuning import Tuning
import pyaudio
import time
import sounddevice as sd
import logging
import webrtcvad
from concurrent.futures import ThreadPoolExecutor
import collections

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the audio recording parameters
CHUNK = 480  # 30ms at 16kHz
CHANNELS = 1  # Record from a single channel
RATE = 16000
RESPEAKER_INDEX = 12  
WAVE_OUTPUT_FOLDER = "audio_clips"
OUTPUT_AUDIO_FILE = os.path.join(WAVE_OUTPUT_FOLDER, "output_audio.wav")

# Maximum buffer size (in seconds)
MAX_BUFFER_SIZE = 20  
MAX_QUEUE_SIZE = 3  # Maximum number of audio chunks in the queue

# Number of processing threads
NUM_PROCESSING_THREADS = 2

# Create the audio clips folder if it doesn't exist
if not os.path.exists(WAVE_OUTPUT_FOLDER):
    os.makedirs(WAVE_OUTPUT_FOLDER)

# Define the SharedDeepSpeech class
class SharedDeepSpeech:
    def __init__(self, model_path):
        logging.info("Loading DeepSpeech model...")
        start_time = time.time()
        self.model = deepspeech.Model(model_path)
        end_time = time.time()
        logging.info(f"Model loaded in {end_time - start_time:.2f} seconds")
        self.lock = threading.Lock()

    def stt(self, audio):
        with self.lock:
            return self.model.stt(audio)

    def sttWithMetadata(self, audio):
        with self.lock:
            return self.model.sttWithMetadata(audio)

# Global variables
model_path = 'deepspeech-0.9.3-models.pbmm'
shared_model = None  # Will be initialized in main()
target_sentence = "sleep"
target_words = target_sentence.split()

# Create queues for the producer-consumer pattern
audio_queue = Queue(maxsize=MAX_QUEUE_SIZE)
word_queue = Queue()
# Create events for controlling the threads
stop_event = threading.Event()
output_ready_event = threading.Event()

def find_respeaker():
    try:
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if dev is None:
            raise ValueError("ReSpeaker not found.")
        return dev
    except usb.core.USBError as e:
        logging.error(f"USB error when finding ReSpeaker: {e}")
        raise

def initialize_tuning(dev):
    try:
        tuning = Tuning(dev)
        
        # Set tuning parameters
        tuning.write('STATNOISEONOFF', 1)
        tuning.write('NONSTATNOISEONOFF', 1)
        tuning.write('GAMMA_NS', 1.5)
        tuning.write('GAMMA_NN', 1.7)
        tuning.write('AGCONOFF', 1)
        tuning.write('AGCDESIREDLEVEL', -23)
        tuning.write('AGCMAXGAIN', 35)
        tuning.write('HPFONOFF', 2)
        tuning.write('ECHOONOFF', 0)
        tuning.write('FREEZEONOFF', 0)

        logging.info("ReSpeaker tuning parameters set successfully")
        return tuning
    except Exception as e:
        logging.error(f"Error initializing Tuning: {e}")
        raise

def is_silent(audio_chunk, threshold=500, silence_percentage=0.9):
    """
    Determine if an audio chunk is silent.
    
    :param audio_chunk: numpy array of audio samples
    :param threshold: amplitude threshold for silence
    :param silence_percentage: percentage of samples below threshold to consider as silence
    :return: boolean indicating if the chunk is silent
    """
    return np.mean(np.abs(audio_chunk) < threshold) > silence_percentage

def record_audio():
    p = pyaudio.PyAudio()
    vad = webrtcvad.Vad(3)  # Set aggressiveness to maximum
    
    # Short-term buffer for VAD
    vad_buffer = collections.deque(maxlen=int(RATE * 0.03))  # 30ms buffer for VAD
    
    # Longer-term buffer for audio processing
    process_buffer = collections.deque(maxlen=int(RATE * 10))  # 10 second buffer for processing
    
    voiced_frames = []
    is_speaking = False
    silent_chunks = 0
    max_silent_chunks = 33  # About 1 second of silence (30ms * 33 ≈ 1s)
    vad_frame_duration = 0.03  # 30ms chunks for WebRTC VAD
    process_frame_duration = 0.5  # 500ms chunks for processing
    vad_samples_per_chunk = int(RATE * vad_frame_duration)
    process_samples_per_chunk = int(RATE * process_frame_duration)

    def audio_callback(in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)[0::6]  # Extract channel 0
        vad_buffer.extend(audio_data)
        process_buffer.extend(audio_data)
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

    try:
        while not stop_event.is_set():
            if len(vad_buffer) >= vad_samples_per_chunk:
                vad_chunk = np.array(list(vad_buffer)[:vad_samples_per_chunk], dtype=np.int16)
                vad_buffer = collections.deque(list(vad_buffer)[vad_samples_per_chunk:], maxlen=int(RATE * 0.03))
                
                try:
                    is_speech = vad.is_speech(vad_chunk.tobytes(), RATE)
                except webrtcvad.Error:
                    logging.warning("WebRTC VAD error, assuming non-speech")
                    is_speech = False

                if is_speech and not is_speaking:
                    is_speaking = True
                    voiced_frames = []
                    silent_chunks = 0

                if is_speaking:
                    if is_speech:
                        silent_chunks = 0
                    else:
                        silent_chunks += 1

                    if silent_chunks > max_silent_chunks or len(process_buffer) >= process_samples_per_chunk:
                        process_chunk = np.array(list(process_buffer)[:process_samples_per_chunk], dtype=np.int16)
                        process_buffer = collections.deque(list(process_buffer)[process_samples_per_chunk:], maxlen=int(RATE * 1))
                        
                        if not is_silent(process_chunk):
                            voiced_frames.extend(process_chunk)

                        if len(voiced_frames) >= RATE * MAX_BUFFER_SIZE or silent_chunks > max_silent_chunks:
                            if len(voiced_frames) > process_samples_per_chunk:
                                audio_data = np.array(voiced_frames, dtype=np.int16)
                                try:
                                    audio_queue.put(audio_data, block=False)
                                    logging.info(f"Putting audio chunk of length {len(audio_data)} into queue (queue size: {audio_queue.qsize()})")
                                except Full:
                                    logging.warning("Audio queue is full. Discarding audio chunk.")
                            is_speaking = False
                            voiced_frames = []
                            silent_chunks = 0

            time.sleep(0.01)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def process_audio_chunk(audio_data):
    try:
        # Perform speech recognition using shared DeepSpeech instance
        text = shared_model.stt(audio_data)
        
        # Check if any target words are detected in the recognized text
        detected_words = []
        for word in text.split():
            if word.lower() in target_words:
                detected_words.append((word.lower(), audio_data))
                logging.info(f"Detected word: {word}")
        
        logging.info(f"Recognized speech: {text}")
        return detected_words
    except deepspeech.DeepSpeechError as e:
        logging.error(f"DeepSpeech error: {e}")
        return []

def process_audio():
    with ThreadPoolExecutor(max_workers=NUM_PROCESSING_THREADS) as executor:
        while not stop_event.is_set():
            try:
                audio_data = audio_queue.get(timeout=1)
                future = executor.submit(process_audio_chunk, audio_data)
                future.add_done_callback(handle_processed_result)
            except Empty:
                continue

def handle_processed_result(future):
    detected_words = future.result()
    for word, audio_data in detected_words:
        word_queue.put((word, audio_data))

def extract_word_audio():
    output_audio = AudioSegment.empty()
    detected_words = []

    while not stop_event.is_set() or not word_queue.empty():
        try:
            word, audio_data = word_queue.get(timeout=1)
            
            if word in detected_words:
                continue
            
            detected_words.append(word)
            
            audio_segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=RATE,
                sample_width=audio_data.dtype.itemsize,
                channels=1
            )
            
            result = shared_model.sttWithMetadata(audio_data)
            timing_info = result.transcripts[0].tokens if result.transcripts else []
            
            word_start_time = None
            word_end_time = None
            
            for i in range(len(timing_info) - len(word) + 1):
                if ''.join(token.text.lower() for token in timing_info[i:i+len(word)]) == word:
                    word_start_time = max(0, timing_info[i].start_time - 0.2)
                    word_end_time = min(len(audio_segment) / 1000, timing_info[i+len(word)-1].start_time + 0.2)
                    break
            
            if word_start_time is not None and word_end_time is not None:
                word_audio_segment = audio_segment[word_start_time * 1000:word_end_time * 1000]
                output_audio += word_audio_segment
                logging.info(f"Extracted audio segment for word '{word}'")
            else:
                logging.warning(f"Word '{word}' not found in the audio segment. Skipping.")

            if all(word in detected_words for word in target_words):
                output_audio.export(OUTPUT_AUDIO_FILE, format="wav")
                logging.info(f"All target words detected. Output audio saved as {OUTPUT_AUDIO_FILE}")
                output_ready_event.set()
                return

        except Empty:
            continue

def main():
    global shared_model
    try:
        # Load the model before starting any threads
        shared_model = SharedDeepSpeech(model_path)

        dev = find_respeaker()
        tuning = initialize_tuning(dev)

        recording_thread = threading.Thread(target=record_audio)
        processing_thread = threading.Thread(target=process_audio)
        extraction_thread = threading.Thread(target=extract_word_audio)

        logging.info("Starting threads...")
        recording_thread.start()
        processing_thread.start()
        extraction_thread.start()

        output_ready_event.wait()  # Wait for all target words to be processed

        # Play the output audio file
        output_audio = AudioSegment.from_wav(OUTPUT_AUDIO_FILE)
        sd.play(output_audio.get_array_of_samples(), output_audio.frame_rate)
        sd.wait()

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping threads...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        stop_event.set()
        recording_thread.join()
        processing_thread.join()
        extraction_thread.join()
        if 'tuning' in locals():
            tuning.close()

if __name__ == "__main__":
    main()