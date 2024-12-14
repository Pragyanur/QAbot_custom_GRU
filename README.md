
# Question Answering Bot, works offline

## Does the following:
- Checks if a face is in front of the camera
- If faces > 0, start interaction with user
- Ends conversation after certain number of questions
- Restarts when detects a human facing the camera

## Uses the following ML/DL models:
- *WHISPER* base for *SPEECH TO TEXT*
- *HAAR CASCADE FRONTAL FACE* model for *FACE DETECTION*
- *CUSTOM TRAINED GRU* model for *TEXT CLASSIFICATION* on custom dataset of questions

## Other tools:
- Python's soundreduce for *NOISE SUPPRESSION*
- Bhashini english Female1 for recorded *RESPONSES*
