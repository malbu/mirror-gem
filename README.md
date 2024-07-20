




![th](https://github.com/malbu/mirror-gem/assets/6825150/2240b314-c817-452d-8f8f-a6ad53575412)

## About

Mirror Gem is an interactive art installation created for 2024 Firefly Arts Festival and was installed at The Cursed Kitchen. This project combines audio processing, speech recognition, and mechanical movement to animate a "gem". The "gem" that appears to be sentient, responding to specific phrases and orienting itself towards speakers. As participants interact with the gem, they uncover a mysterious narrative about an alien entity or consciousness trapped within. But communication is hard...

### Key Features:
- Real-time audio processing and speech recognition
- Servo-controlled movement responding to sound direction
- Interactive storytelling through recognized phrases
- Playback of constructed sentences from recognized words




## Technical Components and Architecture

## Hardware Used

- NVIDIA Jetson Nano 
- ReSpeaker Microphone Array V2
- Adafruit PCA9685 16-Channel Servo Driver
- 

## Software Dependencies
- Jetpack 4.6
- Python 3.7+
- PyAudio
- NumPy
- DeepSpeech 0.9.3 for Jetson Nano
- pydub
- webrtcvad
- Adafruit ServoKit


The system operates on a producer-consumer pattern with multiple threads:
- A recording thread continuously captures audio
- A processing thread handles speech recognition
- An extraction thread manages word identification and sentence reconstruction
- The main thread oversees the overall process and handles servo control



### Key Components:

1. **Hardware Platform**: NVIDIA Jetson Nano
2. **Audio Capture**: PyAudio with ReSpeaker microphone array
3. **Voice Activity Detection**: WebRTC VAD
4. **Speech Recognition**: DeepSpeech 0.9.3
5. **Word and Sentence Processing**: Custom natural language algorithms
6. **Direction of Arrival (DOA)**: ReSpeaker's tuning capabilities
7. **Mechanical Control**: Adafruit ServoKit
8. **Audio Playback**: sounddevice

## ReSpeaker Tuning and Its Importance

Proper tuning of the ReSpeaker, by using Respeaker tuning.py, is essential for the exhibit's responsiveness and interactivity. It directly impacts the installation's ability to understand and respond to participants. On site adjustment is necessary.

---

Created with ❤️  Waiting is HAppening



