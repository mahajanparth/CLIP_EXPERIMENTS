"""Export measured image activations and an offline synchronized stage viewer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from clip_explorer.config import ROOT,MODEL_NAME,MODEL_REVISION
from clip_explorer.data import photo_for_label,open_asset
from clip_explorer.model import image_bytes,encode_text
from clip_explorer.image_activations import inspect_image_payload
from clip_explorer.image_activation_lab import image_document


def main():
    record=photo_for_label('cat');image=open_asset(record);prompt='a photo of a cat'
    trace=inspect_image_payload(image_bytes(image))
    cosine=float(trace['stages']['Normalized CLIP vector'][0]@encode_text([prompt]).numpy()[0])
    target=ROOT/'artifacts'
    (target/'image_activations.html').write_text(image_document(trace,image,record,prompt,cosine))
    np.savez_compressed(target/'image_activations.npz',preprocessed=trace['preprocessed'],**trace['stages'])
    metadata={k:trace[k] for k in ['grid','patch_size','input_shape','hidden_width','output_width']}
    metadata.update(model=MODEL_NAME,model_revision=MODEL_REVISION,image=record,prompt=prompt,cosine=cosine,
        stages={name:list(v.shape) for name,v in trace['stages'].items()},source='Actual frozen image forward pass; temporary hooks; no gradients')
    (target/'image_activations_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Exported real image activations, complete tensors and offline stage viewer.')

if __name__=='__main__':main()
