import io
import re
import sys
import cv2
import time
import wave
import pyaudio
import threading
import torchaudio
import numpy as np
import pandas as pd
from playsound import playsound
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import noisereduce as nr
import soundfile as sf

# Force stdout and stderr to use UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TOTAL_QUESTIONS = 1
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 4
OUTPUT_FILE = "temp_audio.wav"
MAX_SEQ_LEN = 20
CHUNK = 1024  # Number of audio frames per buffer
RATE = 16000  # Sampling rate in Hz
model = WhisperForConditionalGeneration.from_pretrained("whisper_model")
processor = WhisperProcessor.from_pretrained("whisper_model")
classifier = load_model("model_gru.keras")
dataframe = pd.read_csv("dataset.csv")
questions = dataframe["Questions"].tolist()
df = pd.read_csv("Responses_and_Audio.csv")
responses = df["Response"].tolist()
response_paths = df["Path"].tolist()
vocabulary = []
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
begin_conv = False
first =True
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
yellow = (0,255,255)
red = (0,0,255)
green = (0,255,0)
blue = (255,50,100)
frame = []
faces = 0
chat_user = "USER SPEECH TRANSCRIPTION"
chat_bot = "ROBOT RESPONSE"
chat_log = "</>"
dialogues = TOTAL_QUESTIONS


def pre_processing(sentence):
    """pre processing the sentences into words"""
    pattern = r"['\s+.,’!\?\-\[\]\"]"
    words = re.split(pattern, sentence)
    cleaned_words = [word.lower() for word in words if word != ""]
    return cleaned_words

for s in questions:
    vocabulary.extend(pre_processing(s))
vocabulary = list(set(vocabulary))
vocabulary.sort()
tokenizer = Tokenizer(num_words=len(vocabulary) + 1, oov_token="<OOV>")
tokenizer.fit_on_texts(vocabulary)

def predict_class(text):
    """classification into query_type"""
    sequence = tokenizer.texts_to_sequences([text])
    token_seq = pad_sequences(sequence, maxlen=MAX_SEQ_LEN)
    no_zeros = np.count_nonzero(token_seq == 0)
    if no_zeros + 2 >= token_seq.size:
        return -1
    prediction = classifier.predict(token_seq)
    predicted_class = np.argmax(prediction)
    probability = np.max(prediction, axis=1)[0]
    if probability < 0.3:
        return -1
    return predicted_class


def play_response(label):
    """play response sound"""
    global chat_bot
    respond = str(responses[label])
    chat_bot = "Bot: " + respond
    # print(f"Bot: {respond}")
    path = str(response_paths[label])
    playsound(path)


def record_audio(output_file=OUTPUT_FILE, record_seconds=RECORD_SECONDS):
    """capture the sounds for RECORD_SECONDS sec"""
    global chat_log
    chat_log = "Listening..."
    frames = []
    audio = pyaudio.PyAudio()
    # print("Recording...")
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    # print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open("noisy_audio.wav", "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    noisy_audio, sample_rate = sf.read("noisy_audio.wav")
    reduced_noise = nr.reduce_noise(y=noisy_audio, sr=sample_rate)
    sf.write(output_file, reduced_noise, sample_rate)


def process_audio_transformers(path=OUTPUT_FILE):
    """process whisper model using transformers from hugging face"""
    global chat_bot, chat_user, chat_log
    chat_log = "Processing..."
    chat_bot = "o_o"
    audio, sr = torchaudio.load(path)
    inputs = processor(
        audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
    )
    predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    query = transcription[0]
    chat_user = "User: " + str(query)
    # print("User: ", str(query))
    return str(query)


def capture_and_display():
    """capture video and detect front faces for separate thread"""
    global begin_conv, first, frame, faces, chat_user, chat_bot, chat_log, dialogues
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to start camera. Exiting")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # scaleFactor compensates for the smaller faces appearing at a distance
        # minNeighbors tells the  model how conservative it should be while recognizing faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=7, minSize=(50, 50)
        )
        if len(faces) > 0:
            begin_conv = True
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Press Q to quit. Remaining questions = {dialogues}", (20,20), font, font_scale, red, 2, cv2.LINE_8)     
        cv2.putText(frame, chat_log, (20,40), font, font_scale, green, 2, cv2.LINE_8)        
        cv2.putText(frame, chat_user, (20,60), font, font_scale, blue, 2, cv2.LINE_8)        
        cv2.putText(frame, chat_bot, (20,80), font, font_scale, yellow, 2, cv2.LINE_8)  
        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def interaction():
    global begin_conv, first, chat_bot, chat_user, chat_log, dialogues
    while True:
        if first and begin_conv:
            chat_bot = f"Bot: Hi, I am EDUBOT ^_^. You may ask me {TOTAL_QUESTIONS} questions."
            first = False
            playsound("responses/greeting.mp3")
        if dialogues and begin_conv:
            record_audio()
            q_t = process_audio_transformers()
            q_w = pre_processing(q_t)
            q_p = " ".join(q_w)
            label = predict_class(q_p)
            if label < 0 or len(q_w) < 3:
                chat_log = "Please try again"
                # print("Please try again")
                # playsound("responses/try_again.mp3")
                continue
            play_response(label)
            dialogues -= 1
        elif not dialogues and begin_conv:
            chat_bot = "Bot: Thank you for interacting with me"
            playsound("responses/thankyou.mp3")
            begin_conv = False
            first = True
            dialogues = TOTAL_QUESTIONS
            time.sleep(1)
            continue
    

def main():
    """main"""
    # pause_detection = False
    capture_thread = threading.Thread(target=capture_and_display) 
    interaction_thread = threading.Thread(target=interaction)   
    # Start the thread
    capture_thread.start()
    interaction_thread.start()
    capture_thread.join()
    interaction_thread.join()

if __name__ == "__main__":
    main()
