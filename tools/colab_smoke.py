"""Verify NACLIP GPU inference using a synthetic image (not an accuracy test)."""
from pathlib import Path
import os


def main():
    import torch
    from naclip import NACLIP

    os.chdir(Path(__file__).resolve().parents[1])
    assert torch.cuda.is_available(), 'Attach a Colab GPU runtime first.'
    model = NACLIP(clip_path='ViT-B/16', name_path='configs/cls_voc20.txt',
                   pamr_steps=0, slide_crop=0).eval().cuda()
    with torch.inference_mode():
        image = torch.randn(1, 3, 224, 224, device='cuda')
        logits = model.forward_feature(image)
        assert logits.is_cuda and torch.isfinite(logits).all()
        assert logits.shape[-2:] == (224, 224)
        print('GPU inference passed:', tuple(logits.shape), torch.cuda.get_device_name(0))


if __name__ == '__main__':
    # Running a file under tools/ otherwise omits the project root from sys.path.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
