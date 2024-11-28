# data/gcs_loader.py

import torchaudio

from google.cloud import storage
from io import BytesIO


def load_data_from_gcs(bucket_name, audio_folder, text_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    audio_data = []
    text_data = []

    # Load audio files
    audio_blobs = bucket.list_blobs(prefix=audio_folder)
    for blob in audio_blobs:
        if blob.name.endswith(".wav"):
            audio_bytes = BytesIO()
            blob.download_to_file(audio_bytes)
            audio_bytes.seek(0)
            waveform, sample_rate = torchaudio.load(audio_bytes)
            audio_data.append({"audio" : waveform, "sample_rate" : sample_rate})
    
    # Load text files
    text_blobs = bucket.list_blobs(prefix=text_folder)
    for blob in text_blobs:
        if blob.name.endswith(".txt"):
            text_content = blob.download_as_text()
            text_data.append(text_content.strip())

    dataset = [{"audio" : audio, "transcription" : text} for audio, text in zip(audio_data, text_data)]

   # print(f"dataset : {dataset}")

    return dataset 
