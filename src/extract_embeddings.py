
import torch
from transformers import (AutoFeatureExtractor,
                          AutoModel)
from pathlib import Path
import numpy as np
import random
import pandas as pd
import argparse


def set_seed(seed):
    """
    Set random seed
    """
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract_hidden_states(model, extractor, audio_df, num_layers, output_dir):
    """
    Extract hidden states from Wav2Vec 2.0 transformer layers for each input waveform.
    Each layer

    Args:
        audio_df: Pandas DataFrame with columns 'FILENAME', 'AUDIO'.
        num_layers: Total number of transformer layers to extract.

    Returns:
        Each hidden layer is stored in a separate folder; within each folder, each file is stored in a separate .pt file.
        This function only extract activation and store; it does not retrieve.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    layer_dirs = []
    for li in range(1, num_layers + 1):  # hidden_states[0] is embeddings; start at 1
        d = output_root / f"layer_{li:02d}"
        d.mkdir(exist_ok=True)
        layer_dirs.append(d)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model.eval().to(device)

    for idx, (filename, waveform) in enumerate(zip(audio_df['FILENAME'], audio_df['AUDIO'])):
        print(f'Extracting hidden states from waveform {idx + 1}/{len(audio_df)}: {filename}')

        waveform = np.asarray(waveform, dtype="float32") # force audio to be float32.
        # Preprocess waveform
        inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True) 
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            if model.config.model_type == "whisper":
                outputs = model.encoder(**inputs, output_hidden_states=True, output_attentions=False)
            else:
                outputs = model(**inputs, output_hidden_states=True, output_attentions=False)

        hidden_states = outputs.hidden_states  # Tuple of (layer, batch, time, features)
        assert len(hidden_states) == num_layers + 1

        stem = Path(filename).stem

        for li in range(1, num_layers + 1):
            tensor = hidden_states[li].squeeze(0).detach().to("cpu")  
            save_path = layer_dirs[li - 1] / f"{stem}.pt"
            torch.save(tensor, save_path)
               
        
    return {"output_dir": str(output_root), "layers": [f"layer_{i:02d}" for i in range(1, num_layers + 1)]}


def main():
    parser = argparse.ArgumentParser(
        description="Extract deep ASR model hidden states for each recording and save per layer."
    )

    parser.add_argument(
        "--audio-df",
        type=str,
        required=True,
        help="Path to pickle file with columns: FILENAME, AUDIO",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory to store hidden states, one subfolder per layer.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/wav2vec2-base",
        help="Name or path of Wav2Vec2 model (default: facebook/wav2vec2-base).",
    )
    

    args = parser.parse_args()
    set_seed(42)
    model = AutoModel.from_pretrained(args.model)
    extractor = AutoFeatureExtractor.from_pretrained(args.model)

    if model.config.model_type=="whisper":
        num_layers = model.config.encoder_layers
    else:
        num_layers = model.config.num_hidden_layers

    audio_df = pd.read_pickle(args.audio_df)

    # run extraction
    result = extract_hidden_states(
        model=model,
        extractor=extractor,
        audio_df=audio_df,
        num_layers=num_layers,
        output_dir=args.output_dir,
    )

    print("Done. Saved hidden states to:", result["output_dir"])
    print("Layers:", ", ".join(result["layers"]))


if __name__ == "__main__":
    main()