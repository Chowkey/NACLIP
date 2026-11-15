"""Prepare ADE20K and PC59 using NaCLIP_dataset_setup_preprocess.ipynb.

Run on Colab after mounting Google Drive; no model dependencies are needed.
Preserves the notebook's class mapping, official splits and boundary definition.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('/content/drive/MyDrive/datasets'))
    args = parser.parse_args()
    if str(args.data_root.resolve()).startswith('/content/drive/') and not Path('/content/drive/MyDrive').is_dir():
        parser.error('Mount Google Drive before preparing persistent datasets.')

    import hashlib, subprocess

    DATASETS = args.data_root.resolve()
    DOWNLOADS = DATASETS / 'downloads'
    VOC_ROOT = DATASETS / 'VOCdevkit' / 'VOC2010'
    DATASETS.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    # ADE20K archive (skip if already extracted).
    ADE_ZIP = DATASETS / 'ADEChallengeData2016.zip'
    if not (DATASETS / 'ADEChallengeData2016' / 'images' / 'validation').is_dir():
        subprocess.run(['wget', '-nv', '-c', 'https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '-O', str(ADE_ZIP)], check=True)
        subprocess.run(['unzip', '-q', '-o', str(ADE_ZIP), '-d', str(DATASETS)], check=True)

    # VOC2010 images (official host); PASCAL Context reuses these images.
    VOC_TAR = DOWNLOADS / 'VOCtrainval_03-May-2010.tar'
    if not (VOC_ROOT / 'JPEGImages').is_dir() or not (VOC_ROOT / 'ImageSets' / 'Main' / 'val.txt').is_file():
        subprocess.run(['wget', '-nv', '-c', 'https://thor.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar', '-O', str(VOC_TAR)], check=True)
        # Official VOC archives can include VOCdevkit/ in their member paths.
        import tarfile
        with tarfile.open(VOC_TAR) as archive:
            names = [name.lstrip('./') for name in archive.getnames()]
        destination = DATASETS if any(name.startswith('VOCdevkit/') for name in names) else DATASETS / 'VOCdevkit'
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '-xf', str(VOC_TAR), '-C', str(destination)], check=True)

    # Official PASCAL Context train/val annotations: trainval/*.mat and labels.txt.
    PC_TAR = DOWNLOADS / 'pascal_context_trainval.tar.gz'
    if not (VOC_ROOT / '.official_pascal_context_trainval_md5').is_file() or not PC_TAR.is_file():
        subprocess.run(['wget', '-nv', '-c', 'https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz', '-O', str(PC_TAR)], check=True)
    digest = hashlib.md5()
    with PC_TAR.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    md5 = digest.hexdigest()
    assert md5 == 'df034edb2c12aa7d33b42b20bb1796e3', f'Official PASCAL Context archive MD5 mismatch: {md5}'
    marker = VOC_ROOT / '.official_pascal_context_trainval_md5'
    if not marker.is_file():
        VOC_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '-xzf', str(PC_TAR), '-C', str(VOC_ROOT)], check=True)
        marker.write_text(md5, encoding='ascii')
    else:
        assert marker.read_text(encoding='ascii').strip() == md5, 'PASCAL Context extraction marker has unexpected checksum.'
        print('Official PASCAL Context annotations already verified/extracted; skipping extraction.')

    print('VOC2010 / PASCAL Context root:', VOC_ROOT)

    ADE_RAW = DATASETS / 'ADEChallengeData2016'
    PASCAL_VOC2010 = VOC_ROOT
    NACLIP_DIR = Path(__file__).resolve().parents[1]

    ade_img = ADE_RAW / 'images' / 'validation'
    ade_ann = ADE_RAW / 'annotations' / 'validation'
    assert ade_img.is_dir(), f'Missing {ade_img}; check ADEChallengeData2016 layout.'
    assert ade_ann.is_dir(), f'Missing {ade_ann}; check ADEChallengeData2016 layout.'

    ade_images = sorted(ade_img.glob('*.jpg'))
    ade_masks = sorted(ade_ann.glob('*.png'))
    assert len(ade_images) == 2000, f'Expected 2,000 ADE20K validation images; found {len(ade_images)}.'
    mask_names = {p.stem for p in ade_masks}
    missing = [p.name for p in ade_images if p.stem not in mask_names]
    assert not missing, f'{len(missing)} ADE20K images have no matching PNG annotation; e.g. {missing[:5]}'
    print(f'ADE20K val ready: {len(ade_images)} images, {len(ade_masks)} masks')
    print('Use this data_root in cfg_ade20k.py:', ADE_RAW)

    from tqdm.auto import tqdm
    import urllib.request, re
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat

    pc_images = PASCAL_VOC2010 / 'JPEGImages'
    pc_mat_dir = PASCAL_VOC2010 / 'trainval'
    pc_labels_file = PASCAL_VOC2010 / 'labels.txt'
    assert pc_images.is_dir(), f'Missing VOC2010 images: {pc_images}'
    assert pc_mat_dir.is_dir(), f'Missing official PASCAL Context annotations: {pc_mat_dir}'
    assert pc_labels_file.is_file(), f'Missing official labels.txt: {pc_labels_file}'

    def norm_name(name):
        return re.sub(r'[^a-z0-9]', '', name.lower())

    # Verify the official PC59 class set, but serialize masks in NACLIP's class order.
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    pc59_names_file = DOWNLOADS / '59_labels.txt'
    if not pc59_names_file.is_file():
        urllib.request.urlretrieve('https://cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt', pc59_names_file)
    naclip_classes = [s.strip() for s in (NACLIP_DIR / 'configs' / 'cls_context59.txt').read_text(encoding='utf-8').splitlines() if s.strip()]
    official_pc59 = [line.split(':', 1)[1].strip() for line in pc59_names_file.read_text(encoding='utf-8').splitlines() if ':' in line]
    assert len(naclip_classes) == 59 and len(official_pc59) == 59
    assert {norm_name(x) for x in naclip_classes} == {norm_name(x) for x in official_pc59}, 'NACLIP classes differ from official PC59 list'

    # Build raw LabelMap IDs from the official full label-name file.
    raw_id_by_name = {}
    for line in pc_labels_file.read_text(encoding='utf-8').splitlines():
        if ':' in line:
            idx, name = line.split(':', 1)
            raw_id_by_name[norm_name(name)] = int(idx.strip())
    source_ids = [raw_id_by_name[norm_name(name)] for name in naclip_classes]
    assert len(set(source_ids)) == 59, 'Duplicate raw category IDs in NACLIP PC59 mapping'

    # Use the official VOC2010 validation split; standard PASCAL Context reports 4,998 train / 5,105 val.
    val_list_path = PASCAL_VOC2010 / 'ImageSets' / 'Main' / 'val.txt'
    assert val_list_path.is_file(), f'Missing official VOC2010 validation split: {val_list_path}'
    val_ids = [line.strip() for line in val_list_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(val_ids) == 5105 and len(set(val_ids)) == 5105, f'Expected 5,105 unique official VOC2010 val IDs; found {len(val_ids)} rows / {len(set(val_ids))} unique.'

    pc_split = PASCAL_VOC2010 / 'ImageSets' / 'SegmentationContext' / 'val.txt'
    pc_split.parent.mkdir(parents=True, exist_ok=True)
    pc_split.write_text('\n'.join(val_ids) + '\n', encoding='utf-8')
    pc_seg = PASCAL_VOC2010 / 'SegmentationClassContext'
    pc_seg.mkdir(parents=True, exist_ok=True)
    missing_img = [i for i in val_ids if not (pc_images / f'{i}.jpg').is_file()]
    missing_mat = [i for i in val_ids if not (pc_mat_dir / f'{i}.mat').is_file()]
    assert not missing_img, f'Missing validation images: {missing_img[:5]}'
    assert not missing_mat, f'Missing official PASCAL Context .mat annotations: {missing_mat[:5]}'

    # 0=background; class IDs 1..59 follow NACLIP cls_context59.txt; unselected classes=255.
    for image_id in tqdm(val_ids, desc='Convert official PASCAL Context val to PC59', mininterval=5):
        mat = loadmat(pc_mat_dir / f'{image_id}.mat')
        if 'LabelMap' not in mat:
            raise KeyError(f'No LabelMap in {image_id}.mat')
        raw = np.asarray(mat['LabelMap'])
        if raw.ndim != 2:
            raise ValueError(f'Expected 2-D LabelMap for {image_id}, got {raw.shape}')
        pc59 = np.full(raw.shape, 255, dtype=np.uint8)
        pc59[raw == 0] = 0
        for class_index, raw_id in enumerate(source_ids, start=1):
            pc59[raw == raw_id] = class_index
        Image.fromarray(pc59).save(pc_seg / f'{image_id}.png')

    for image_id in val_ids[:30]:
        arr = np.asarray(Image.open(pc_seg / f'{image_id}.png'))
        assert arr.ndim == 2 and set(np.unique(arr)) <= set(range(60)) | {255}
    print(f'PASCAL Context PC59 val ready: {len(val_ids)} official annotations converted in NACLIP class order.')
    print('Masks:', pc_seg)
    print('Split:', pc_split)

    from PIL import Image
    import numpy as np

    pc_split = PASCAL_VOC2010 / 'ImageSets' / 'SegmentationContext' / 'val.txt'
    pc_seg = PASCAL_VOC2010 / 'SegmentationClassContext'
    assert pc_split.is_file() and pc_seg.is_dir()
    check_ids = [line.strip() for line in pc_split.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(check_ids) == 5105 and len(set(check_ids)) == 5105
    missing_img = [i for i in check_ids if not (pc_images / f'{i}.jpg').is_file()]
    missing_mask = [i for i in check_ids if not (pc_seg / f'{i}.png').is_file()]
    assert not missing_img, f'Missing images: {missing_img[:5]}'
    assert not missing_mask, f'Missing converted official PC59 masks: {missing_mask[:5]}'
    for image_id in check_ids[:30]:
        arr = np.asarray(Image.open(pc_seg / f'{image_id}.png'))
        assert arr.ndim == 2 and set(np.unique(arr)) <= set(range(60)) | {255}
    print(f'Official PASCAL Context PC59 ready: {len(check_ids)} val images, images/masks complete.')
    print('Use this data_root in cfg_context59.py:', PASCAL_VOC2010)

    from scipy.ndimage import distance_transform_edt
    from tqdm.auto import tqdm
    import numpy as np
    from PIL import Image

    PREPROCESS_ROOT = DATASETS / 'preprocessed'
    BOUNDARY_RADIUS_PX = 3.0
    IGNORE_IDS = {0, 255}

    def make_boundary_and_valid(label_array, radius_px=3.0, ignore_ids=(0, 255)):
        labels = np.asarray(label_array)
        if labels.ndim != 2:
            raise ValueError(f'Expected a 2-D class-index mask; got {labels.shape}')
        valid = np.ones(labels.shape, dtype=bool)
        for ignore_id in ignore_ids:
            valid &= labels != ignore_id
        seeds = np.zeros(labels.shape, dtype=bool)
        h_diff = (labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
        seeds[:, :-1] |= h_diff
        seeds[:, 1:] |= h_diff
        v_diff = (labels[:-1, :] != labels[1:, :]) & valid[:-1, :] & valid[1:, :]
        seeds[:-1, :] |= v_diff
        seeds[1:, :] |= v_diff
        band = (distance_transform_edt(~seeds) <= radius_px) & valid if seeds.any() else np.zeros(labels.shape, dtype=bool)
        return band.astype(np.uint8) * 255, valid.astype(np.uint8) * 255

    def process_split(dataset_name, ids, label_dir):
        boundary_dir = PREPROCESS_ROOT / 'boundary_3px' / dataset_name
        valid_dir = PREPROCESS_ROOT / 'valid_pixels' / dataset_name
        boundary_dir.mkdir(parents=True, exist_ok=True)
        valid_dir.mkdir(parents=True, exist_ok=True)
        for image_id in tqdm(ids, desc=f'Boundary preprocess {dataset_name}', mininterval=5):
            path = label_dir / f'{image_id}.png'
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as im:
                labels = np.asarray(im)
            boundary, valid = make_boundary_and_valid(labels, BOUNDARY_RADIUS_PX, IGNORE_IDS)
            Image.fromarray(boundary).save(boundary_dir / f'{image_id}.png')
            Image.fromarray(valid).save(valid_dir / f'{image_id}.png')
        return {'images': len(ids), 'boundary_dir': str(boundary_dir), 'valid_pixels_dir': str(valid_dir)}

    ade_img_dir = ADE_RAW / 'images' / 'validation'
    ade_ann_dir = ADE_RAW / 'annotations' / 'validation'
    ade_ids = sorted(p.stem for p in ade_img_dir.glob('*.jpg'))
    assert len(ade_ids) == 2000, f'Expected 2,000 ADE20K validation images; found {len(ade_ids)}'
    ade_result = process_split('ade20k_150_val', ade_ids, ade_ann_dir)
    pc_result = process_split('pascal_context_59_val', val_ids, pc_seg)
    print({'ade20k_150_val': ade_result, 'pascal_context_59_val': pc_result})

    import json
    manifest = {
        'ade20k_150': {'split': 'validation', 'images': len(ade_ids), 'data_root': str(ADE_RAW)},
        'pascal_context_59': {
            'split': 'validation', 'images': len(val_ids), 'data_root': str(PASCAL_VOC2010),
            'annotation_source': 'official Stanford PASCAL Context trainval.tar.gz (trainval/*.mat + labels.txt)',
            'class_order_source': 'NACLIP configs/cls_context59.txt; checked against official 59_labels.txt',
            'split_source': 'official VOC2010 ImageSets/Main/val.txt',
            'split_count_note': 'Official VOC2010 val split contains 5,105 IDs; PASCAL Context uses these same images.',
            'label_ids': {'background': 0, 'semantic_classes': '1..59 in NACLIP class order', 'ignored': 255},
            'background_excluded_at_eval': True,
        },
        'preprocessing': {
            'boundary_radius_px': BOUNDARY_RADIUS_PX,
            'boundary_definition': 'valid horizontal/vertical class-transition pixels, Euclidean distance <= 3 px, intersected with valid pixels',
            'ignore_ids': sorted(IGNORE_IDS), 'resize_applied': False,
            'ade20k_150_val': ade_result, 'pascal_context_59_val': pc_result,
        },
    }
    manifest_path = DATASETS / 'naclip_validation_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(manifest_path.read_text())


if __name__ == '__main__':
    main()
