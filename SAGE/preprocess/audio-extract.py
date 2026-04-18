import os
import whisper
import json
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("turbo").to(device)

def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    try:
        result = model.transcribe(audio_path)
        transcript = result["text"]
        return transcript.strip()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "ERROR"

def transcribe_from_file(folder_path: str, extensions=None) -> dict:
    if extensions is None:
        extensions = [".wav", ".mp3", ".m4a", ".flac", ".aac"]

    transcripts = {}
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            audio_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename} ...")
            text = transcribe_audio(audio_path)
            transcripts[filename] = text

    return transcripts

def save_transcripts(transcripts: dict, save_path: str):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=4, ensure_ascii=False)
    print(f"Transcripts saved successfully to: {save_path}")

if __name__ == "__main__":
    folder = r"audio_folder"                                       # To Do
    save_path = r'save_path.json'                                  # To Do
    results = transcribe_from_file(folder)
    save_transcripts(results, save_path)
