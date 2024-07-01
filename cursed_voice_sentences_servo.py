
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
from logging.handlers import QueueHandler, QueueListener
import queue
import webrtcvad
from concurrent.futures import ThreadPoolExecutor
import collections
import wave
import time
from adafruit_servokit import ServoKit
import random
import struct

# Set the audio recording parameters
CHUNK = 480  # 30ms at 16kHz
CHANNELS = 1  # Record from a single channel
RATE = 16000
RESPEAKER_INDEX = 11  
WAVE_OUTPUT_FOLDER = "audio_clips"
WORD_AUDIO_FOLDER = os.path.join(WAVE_OUTPUT_FOLDER, "word_audio")

# Maximum buffer size (in seconds)
MAX_BUFFER_SIZE = 20  
MAX_QUEUE_SIZE = 3  # Maximum number of audio chunks in the queue

# Number of processing threads
NUM_PROCESSING_THREADS = 2

# Create the necessary folders if they don't exist
for folder in [WAVE_OUTPUT_FOLDER, WORD_AUDIO_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Constants for more visible logging
RECOGNIZED_WORD_FORMAT = "\n{}\nRECOGNIZED WORD: {}\n{}\n".format('*' * 20, '{}', '*' * 20)
QUEUE_STATUS_FORMAT = "\n{}\nQUEUE STATUS:\nAudio Queue: {}/{}\nWord Queue: {}\nWords Found: {}/{}\n{}\n".format('=' * 40, '{}', '{}', '{}', '{}', '{}', '=' * 40)

# Global variables
model_path = 'deepspeech-0.9.3-models.pbmm'
shared_model = None
target_sentences = [
    "wake up computer",
    "go to sleep",
    "what's the weather like",
    "set an alarm for tomorrow"
]
all_target_words = set()
found_words = {}

# Create queues for the producer-consumer pattern
audio_queue = Queue(maxsize=MAX_QUEUE_SIZE)
word_queue = Queue()

# Create events for controlling the threads
stop_event = threading.Event()
output_ready_event = threading.Event()

# Initialize the servo
pca = ServoKit(channels=16)
servo = pca.servo[0]

# Global variable to store the current DOA
current_doa = 0

def setup_logging():
    log_queue = queue.Queue(-1)  # No limit on size
    queue_handler = QueueHandler(log_queue)
    file_handler = logging.FileHandler('audio_processing.log')
    console_handler = logging.StreamHandler()
    listener = QueueListener(log_queue, file_handler, console_handler)
    root = logging.getLogger()
    root.addHandler(queue_handler)
    root.setLevel(logging.INFO)
    return listener

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
    samples_below_threshold = np.sum(np.abs(audio_chunk) < threshold)
    total_samples = len(audio_chunk)
    actual_silence_percentage = samples_below_threshold / total_samples
    
    is_silent = actual_silence_percentage > silence_percentage
    
    logging.info(f"Audio chunk analysis: "
                 f"threshold={threshold}, "
                 f"silence_percentage={silence_percentage:.2f}, "
                 f"actual_silence_percentage={actual_silence_percentage:.2f}, "
                 f"is_silent={is_silent}")
    
    return is_silent

def log_queue_status():
    if logging.getLogger().isEnabledFor(logging.INFO):
        logging.info(QUEUE_STATUS_FORMAT.format(
            audio_queue.qsize(), audio_queue.maxsize, word_queue.qsize(),
            len(found_words), len(all_target_words)
        ))

def record_audio():
    p = pyaudio.PyAudio()
    vad = webrtcvad.Vad(3)  # Set aggressiveness to maximum
    
    vad_buffer = collections.deque(maxlen=int(RATE * 0.03))  # 30ms buffer for VAD
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
                        
                        chunk_is_silent = is_silent(process_chunk)
                        if not chunk_is_silent:
                            voiced_frames.extend(process_chunk)
                            logging.info(f"Non-silent chunk detected, length: {len(process_chunk)}")
                        else:
                            logging.info(f"Silent chunk detected, length: {len(process_chunk)}")

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
        # Perform silence detection
        chunk_is_silent = is_silent(audio_data)
        
        if not chunk_is_silent:
            # Perform speech recognition using shared DeepSpeech instance
            text = shared_model.stt(audio_data)
            
            # Check if any target words are detected in the recognized text
            detected_words = []
            for word in text.split():
                if word.lower() in all_target_words and word.lower() not in found_words:
                    detected_words.append((word.lower(), audio_data))
                    logging.info(RECOGNIZED_WORD_FORMAT.format(word))
            
            if detected_words:
                logging.info(f"Recognized speech: {text}")
                log_queue_status()
            return detected_words
        else:
            logging.debug("Silent chunk detected, skipping speech recognition")
            return []
    except deepspeech.DeepSpeechError as e:
        logging.error(f"DeepSpeech error: {e}")
        return []

def handle_processed_result(future):
    detected_words = future.result()
    for word, audio_data in detected_words:
        try:
            word_queue.put((word, audio_data), block=False)
            logging.info(f"Added word '{word}' to word queue")
        except Full:
            logging.warning(f"Word queue is full. Discarding word: {word}")
    if detected_words:
        log_queue_status()

def process_audio():
    with ThreadPoolExecutor(max_workers=NUM_PROCESSING_THREADS) as executor:
        while not stop_event.is_set():
            try:
                audio_data = audio_queue.get(timeout=1)
                logging.debug(f"Processing audio chunk of length {len(audio_data)}")
                future = executor.submit(process_audio_chunk, audio_data)
                future.add_done_callback(handle_processed_result)
            except Empty:
                pass

def save_word_audio(word, audio_data):
    filename = os.path.join(WORD_AUDIO_FOLDER, f"{word}_{time.time()}.wav")
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes per sample for int16
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())
    logging.info(f"Saved original audio for word '{word}' to {filename}")

def detect_voice_activity_webrtc(vad, duration=0.3, num_checks=10):
    """
    Detect voice activity using WebRTC VAD.
    
    :param vad: WebRTC VAD object
    :param duration: Duration of each audio chunk to check (in seconds)
    :param num_checks: Number of consecutive chunks to check
    :return: True if voice activity is detected, False otherwise
    """
    chunk_samples = int(RATE * duration)
    for _ in range(num_checks):
        audio_chunk = sd.rec(chunk_samples, samplerate=RATE, channels=1, dtype='int16')
        sd.wait()
        is_speech = vad.is_speech(audio_chunk.tobytes(), RATE)
        if is_speech:
            logging.info("Voice activity detected by WebRTC VAD")
            return True
        time.sleep(0.1)  # Short sleep between checks
    logging.info("No voice activity detected by WebRTC VAD")
    return False

def play_audio_twice(audio_segment):
    """
    Play the given audio segment twice with a short pause in between.
    
    :param audio_segment: pydub.AudioSegment to play
    """
    audio_array = np.array(audio_segment.get_array_of_samples())
    sd.play(audio_array, audio_segment.frame_rate)
    sd.wait()
    time.sleep(0.5)  # Half-second pause between plays
    sd.play(audio_array, audio_segment.frame_rate)
    sd.wait()

def update_doa():
    global current_doa
    dev = find_respeaker()
    tuning = Tuning(dev)
    current_doa = tuning.direction
    tuning.close()

def move_servo(angle):
    servo.angle = angle

def extract_word_audio():
    last_log_time = time.time()
    last_doa_check_time = time.time()
    vad = webrtcvad.Vad(3)  # Create a new VAD object with aggressiveness level 3
    while not stop_event.is_set() or not word_queue.empty():
        try:
            word, audio_data = word_queue.get(timeout=1)
            
            if word in found_words:
                logging.info(f"Word '{word}' already processed. Skipping.")
                continue
            
            logging.info(RECOGNIZED_WORD_FORMAT.format(word))
            
            # Save original audio of the found word
            save_word_audio(word, audio_data)
            
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
                found_words[word] = word_audio_segment
                logging.info(f"Extracted audio segment for word '{word}' (duration: {word_audio_segment.duration_seconds:.2f}s)")
                logging.info(f"Words found: {len(found_words)}/{len(all_target_words)}")
            else:
                logging.warning(f"Word '{word}' not found in the audio segment. Skipping.")

            # Check if any target sentences are complete
            complete_sentences = [
                sentence for sentence in target_sentences
                if all(word in found_words for word in sentence.split())
            ]

            for sentence in complete_sentences:
                output_audio = AudioSegment.empty()
                for word in sentence.split():
                    output_audio += found_words[word]
                
                output_filename = os.path.join(WAVE_OUTPUT_FOLDER, f"{sentence.replace(' ', '_')}.wav")
                output_audio.export(output_filename, format="wav")
                logging.info(f"Target sentence '{sentence}' completed. Output audio saved as {output_filename}")
                
                # Update DOA and move servo before playing
                update_doa()
                move_servo(current_doa)
                logging.info(f"Moved servo to angle: {current_doa}")

                # Wait for voice activity before playing
#                logging.info("Waiting for voice activity to play the sentence...")
#                while not detect_voice_activity_webrtc(vad):
#                    time.sleep(0.5)
#                    if stop_event.is_set():
#                        return

                logging.info("Voice activity detected. Playing sentence twice.")
                play_audio_twice(output_audio)
                
                target_sentences.remove(sentence)

            if not target_sentences:
                logging.info("All target sentences have been processed. Stopping.")
                output_ready_event.set()
                return

            # Check if 3 minutes have passed since last DOA check
            current_time = time.time()
            if current_time - last_doa_check_time > 180:  # 3 minutes = 180 seconds
                if random.random() < 0.5:  # 50% chance
                    update_doa()
                    move_servo(current_doa)
                    logging.info(f"Random DOA check: Moved servo to angle: {current_doa}")
                last_doa_check_time = current_time

            # Log queue status every 30 seconds or when a word is found
            if current_time - last_log_time > 30:
                log_queue_status()
                last_log_time = current_time

        except Empty:
            pass
        except Exception as e:
            logging.error(f"Error in extract_word_audio: {e}")

def main():
    global shared_model, all_target_words
    log_listener = setup_logging()
    log_listener.start()
    try:
        # Initialize all_target_words
        all_target_words = set(word for sentence in target_sentences for word in sentence.split())

        # Load the model before starting any threads
        shared_model = SharedDeepSpeech(model_path)

        dev = find_respeaker()
        tuning = initialize_tuning(dev)

        # Initialize servo
        servo.angle = 0  # Set initial position

        recording_thread = threading.Thread(target=record_audio)
        processing_thread = threading.Thread(target=process_audio)
        extraction_thread = threading.Thread(target=extract_word_audio)

        logging.info("Starting threads...")
        recording_thread.start()
        processing_thread.start()
        extraction_thread.start()

        output_ready_event.wait()  # Wait for all target sentences to be processed

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
        log_listener.stop()

if __name__ == "__main__":
    main()
