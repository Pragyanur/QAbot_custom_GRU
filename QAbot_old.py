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


# Force stdout and stderr to use UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

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


def pre_processing(sentence):
    """pre processing the sentences into words"""
    pattern = r"['\s+.,’!\?\-\[\]\"]"
    words = re.split(pattern, sentence)
    cleaned_words = [word.lower() for word in words if word != ""]
    return cleaned_words


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
    path = str(response_paths[label])
    playsound(path)
    print(f"Bot: {str(responses[label])}")
    # time.sleep(3)


def record_audio(output_file=OUTPUT_FILE, record_seconds=RECORD_SECONDS):
    """capture the sounds for RECORD_SECONDS sec"""
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
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


def process_audio_transformers(path=OUTPUT_FILE):
    """process whisper model using transformers from hugging face"""
    audio, sr = torchaudio.load(path)
    inputs = processor(
        audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
    )
    predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    query = transcription[0]
    print("user: ", str(query))
    return str(query)


def capture_and_display():
    global begin_conv, first, frame, faces
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        cv2.putText(frame, "Press Q to quit", (20,20), font, font_scale, color, 2, cv2.LINE_AA)        

        if len(faces) > 0:
            begin_conv = True
            for i,(x, y, w, h) in enumerate(faces):
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"face {i + 1}", (x - 5, y - 5), font, font_scale, color, 2, cv2.LINE_AA)

        cv2.imshow("Video Stream", frame)
        
        # Press 'q' to quit the display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# def user_interacting():
#     """conversation started function"""
#     print("User detected... Conversation started...")
#     dialogues = 5
#     retry = 0
#     playsound("responses/greeting.mp3")
#     print("Bot: You may ask me general knowledge, or questions related to Science.")
#     while dialogues:
#         if retry > 5:
#             print(
#                 "Bot: You may ask me general knowledge, or questions related to Science."
#             )
#             retry = 0
#         record_audio()
#         query_text = process_audio_transformers()
#         query_words = pre_processing(query_text)
#         query_processed = " ".join(query_words)
#         label = predict_class(query_processed)
#         if label < 0 or len(query_words) < 3:
#             retry += 1
#             print("Bot: Please try again")
#             continue
#         play_response(label)
#         retry = 0
#         dialogues -= 1
#     playsound("responses/thankyou.mp3")
#     print("Conversation stopped")


for s in questions:
    vocabulary.extend(pre_processing(s))
vocabulary = list(set(vocabulary))
vocabulary.sort()
tokenizer = Tokenizer(num_words=len(vocabulary) + 1, oov_token="<OOV>")
tokenizer.fit_on_texts(vocabulary)


begin_conv = False
first =True
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
color = (0,255,255)
frame = []
faces = 0

def interaction():
    global begin_conv, first
    dialogues = 0
    while True:
        if first and begin_conv:
            playsound("responses/greeting.mp3")
            dialogues = 5
            first = False
        if dialogues and begin_conv:
            record_audio()
            q_t = process_audio_transformers()
            q_w = pre_processing(q_t)
            q_p = " ".join(q_w)
            label =predict_class(q_p)
            if label < 0 or len(q_w) < 3:
                print("Bot: Please try again")
                # playsound("responses/try_again.mp3")
                continue
            play_response(label)
            dialogues -= 1
            if dialogues < 1:
                playsound("responses/thankyou.mp3")
                begin_conv = False
                first = True
                break
    time.sleep(1)

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




    # while True:
    #     if not pause_detection:
    #         cap = cv2.VideoCapture(0)
    #         ret, frame = cap.read()
    #         if not ret:
    #             print("Failed to grab frame. Exitting")
    #             break
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         faces = face_cascade.detectMultiScale(
    #             gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    #         )
    #         if len(faces) > 0:
    #             # Pause further detection
    #             pause_detection = True
    #             user_interacting()
    #             cap.release()
    #             cv2.destroyAllWindows()
    #             pause_detection = False
    #     else:
    #         time.sleep(1)
        


if __name__ == "__main__":
    main()
