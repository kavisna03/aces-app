import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import shutil
import numpy as np
import torch
from moviepy.editor import VideoFileClip
import torchaudio
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
import tensorflow as tf


def process_and_transcribe_video(video_path):
    """
    Processes a video file, extracts audio, and transcribes it.

    Args:
        video_path (str): Path to the video file.

    Returns:
        str: The transcription of the video's audio.
    """

    output_dir = 'extracted_audios'
    video_path = os.path.abspath(video_path)

    # Ensure GPU is used if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_dtype = torch.float32
    model_id = "openai/whisper-large-v3"

    # Load Whisper model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Set up the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        device=device,
    )

    # Extract audio from the video
    def extract_audio(video_path):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + ".wav")

        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
        clip.close()
        return audio_path

    # Resample audio to the required sample rate
    def resample_audio(audio_path, output_sample_rate=16000):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != output_sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=output_sample_rate)
        return waveform

    # Transcribe the audio
    def transcribe(audio_path):
        waveform = resample_audio(audio_path)
        result = pipe(waveform.numpy()[0], generate_kwargs={"language": "en", "task": "transcribe"})
        return result["text"]

    # Processing the video
    audio_path = ""

    try:
        print(f"Processing video: {video_path}")
        audio_path = extract_audio(video_path)
        print(f"Audio extracted to: {audio_path}")
        transcription = transcribe(audio_path)
        print("Transcription has been extracted.")

        return transcription
    except Exception as e:
        print(f"Error processing video: {e}")
        return None
    finally:
        if audio_path:
            os.remove(audio_path)

        shutil.rmtree(output_dir)


# Tokenization function
def tokenize_text(text, tokenizer):
    # Define constants
    MODEL_NAME = 'distilbert-base-cased'

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    MAX_LENGTH = determine_max_length(text, tokenizer)

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True
    )
    input_ids = np.asarray([inputs['input_ids']], dtype='int32')
    attention_mask = np.asarray([inputs['attention_mask']], dtype='int32')
    return input_ids, attention_mask


def determine_max_length(transcription, tokenizer, max_limit=512):
    try:
        input_token_len = len(tokenizer.tokenize(transcription))
        print(f"The number of tokens in the input is {input_token_len}")
    except Exception as e:
        print(f"Error tokenizing the input: {e}")
        return max_limit  # Default to max limit in case of error

    # Adding 2 tokens for [CLS] and [SEP], and capping at max_limit
    max_length = min(input_token_len + 2, max_limit)
    print(f"MAX_LENGTH set to {max_length}")
    return max_length


# Define class to create a custom DistilBERT embedding layer
class DistilBertEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, distilbert_model):
        super(DistilBertEmbeddingLayer, self).__init__()
        self.distilbert = distilbert_model

    def create_embeddings(self, inputs):
        input_ids, attention_mask = inputs
        embeddings = self.distilbert(input_ids, attention_mask=attention_mask)[0]
        return embeddings


# Build model function
def build_model(transcription):
    # Define constants
    MODEL_NAME = 'distilbert-base-cased'

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    MAX_LENGTH = determine_max_length(transcription, tokenizer)

    config = DistilBertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True, output_attentions=True)
    distilbert = TFDistilBertModel.from_pretrained(MODEL_NAME, config=config)

    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), name='input_token', dtype='int32')
    attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), name='masked_token', dtype='int32')

    embeddings_layer = DistilBertEmbeddingLayer(distilbert)
    embeddings = embeddings_layer.create_embeddings([input_ids, attention_mask])

    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(embeddings)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    output = tf.keras.layers.Dense(4, activation='softmax')(X)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # Ensure the model is created properly
    model.summary()

    return model


def predict_grade_level(transcription, model):
    # Define constants
    MODEL_NAME = 'distilbert-base-cased'
    LABELS = ['Elementary School (Grades 1-5)', 'High School (Grades 9-12)', 'Kindergarten', 'Middle School (Grades 6-8)']

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize the input text
    input_ids, attention_mask = tokenize_text(transcription, tokenizer)

    # Predict
    predictions = model.predict([input_ids, attention_mask])
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = LABELS[predicted_class]

    return predicted_label