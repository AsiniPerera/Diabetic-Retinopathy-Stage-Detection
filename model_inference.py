"""Run the selected EfficientNetB0 checkpoint with notebook preprocessing and routing."""
from pathlib import Path
import base64
import hashlib
from io import BytesIO
import json
import tempfile
import threading
import time
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b0

from model_processing import PreprocessConfig, read_rgb, extract_retina, resize_pad, reconstruct_rgb, assess_upload_quality, QUALITY_GATE_CONFIG
from retinal_identity import assess_retinal_appearance

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'section5_single_aug_weighted_v01'
NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
CONFIG = PreprocessConfig()
LOCK = threading.Lock()


def preview_data_url(rgb, size=256):
    image = Image.fromarray(rgb)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=78, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'


class TrainedPredictor:
    def __init__(self):
        identity = json.loads((RUN / 'run_identity.json').read_text(encoding='utf-8'))
        selection = json.loads((RUN / 'model_selection.json').read_text(encoding='utf-8'))
        self.routing = json.loads((RUN / 'section9_prototype/routing_configuration.json').read_text(encoding='utf-8'))
        if asdict(QUALITY_GATE_CONFIG) != self.routing['quality_configuration']:
            raise RuntimeError('Quality gate differs from the saved routing calibration.')
        checkpoint_path = RUN / 'selected_model.pt'
        self.checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if self.checkpoint_hash != self.routing['checkpoint_sha256']:
            raise RuntimeError('The selected checkpoint does not match the saved calibration.')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if checkpoint['run_hash'] != run_hash or checkpoint['selection'] != selection:
            raise RuntimeError('Checkpoint and training identity do not match.')
        torch.set_num_threads(4)
        self.model = efficientnet_b0(weights=None)
        features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(nn.Dropout(float(identity['config']['dropout'])), nn.Linear(features, 5))
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
        self.std = torch.tensor([.229, .224, .225]).view(3, 1, 1)

    def predict(self, payload, filename):
        started = time.perf_counter()
        if not payload or len(payload) > CONFIG.max_file_mb * 1024**2:
            raise ValueError('Choose a JPEG or PNG image under 128 MB.')
        processing_log = []

        def record_step(number, title, description, step_started, details=None, status='complete'):
            processing_log.append({
                'number': number,
                'title': title,
                'description': description,
                'status': status,
                'duration_ms': round((time.perf_counter() - step_started) * 1000, 2),
                'details': details or {},
            })

        def skip_remaining(first_number, reason):
            titles = [
                ('Retinal-mask extraction and crop', 'Detect the retina, form its mask, and crop to the detected bounding box.'),
                ('Crop geometry recorded', 'Record source dimensions, ROI bounds and crop dimensions for the audit log.'),
                ('Aspect-preserving resize', 'Resize the crop to the model canvas and pad without distorting its shape.'),
                ('RGB reconstruction', 'Apply the selected inference preprocessing branch and mask the outside region.'),
                ('Channel layout', 'Convert the image from height-width-channel to channel-height-width.'),
                ('Pixel scaling', 'Convert 8-bit RGB pixel values to the [0, 1] range.'),
                ('Backbone normalization', 'Normalize the input with the EfficientNet ImageNet channel statistics.'),
                ('Model input validation', 'Check the tensor shape and data type before inference.'),
                ('Quality-gate routing', 'Decide whether the image may proceed to model inference.'),
                ('EfficientNetB0 feature extraction', 'Run the prepared image through the trained feature backbone.'),
                ('Calibrated dropout inference', 'Run the saved dropout sampling and temperature-scaling procedure.'),
                ('Prediction and review decision', 'Aggregate class probabilities and apply review thresholds.'),
            ]
            for number, (title, description) in enumerate(titles, start=first_number):
                record_step(number, title, description, time.perf_counter(),
                            {'reason': reason}, status='skipped')

        step_started = time.perf_counter()
        upload_bytes = len(payload)
        record_step(1, 'Upload received', 'Receive the selected image bytes for local inference.',
                    step_started, {'filename': Path(filename).name, 'file_bytes': upload_bytes})

        step_started = time.perf_counter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'upload'
            path.write_bytes(payload)
            rgb, image_info = read_rgb(path, CONFIG)
        record_step(2, 'Image validation and decoding',
                    'Validate the JPEG/PNG, apply EXIF orientation and decode RGB pixels.',
                    step_started, {
                        'format': image_info['format'],
                        'dimensions': f"{image_info['validated_width']} × {image_info['validated_height']}",
                        'channels': image_info['channels'],
                        'dtype': image_info['dtype'],
                        'orientation': image_info['orientation_policy'],
                        'transparency composited': image_info['transparency_composited'],
                    })

        step_started = time.perf_counter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'upload'
            path.write_bytes(payload)
            quality = assess_upload_quality(path)
        identity = assess_retinal_appearance(rgb)
        record_step(3, 'Image quality assessment',
                    'Measure image quality and check for fundus photograph appearance.',
                    step_started, {
                        'decision': quality['quality_status'],
                        **quality['metrics'],
                        'quality method': quality['method'],
                        'fundus photo check': 'passed' if identity['likely_retinal'] else 'failed',
                    }, status='rejected' if not identity['likely_retinal'] else
                    'review' if quality['human_review_required'] else
                    'rejected' if not quality['classification_permitted'] else 'complete')
        visual_steps = [{
            'title': 'Validated upload',
            'description': 'EXIF orientation corrected and image decoded to RGB before quality checks.',
            'image': preview_data_url(rgb),
        }]
        result = {'filename': filename, 'model': 'EfficientNetB0', 'quality': quality,
                  'retinal_appearance': {'likely_retinal': identity['likely_retinal']},
                  'checkpoint_sha256': self.checkpoint_hash, 'reasons': list(quality['reasons']),
                  'processing_log': processing_log, 'visual_steps': visual_steps}
        if not quality['classification_permitted']:
            skip_remaining(4, 'The quality gate rejected this image; no model prediction was run.')
            result.update(status='quality_rejected', classification_permitted=False)
            return result
        if not identity['likely_retinal']:
            skip_remaining(4, 'The image did not pass the fundus photograph appearance check; no model prediction was run.')
            result.update(status='non_retinal', classification_permitted=False,
                          reasons=['The image could not be verified as a colour fundus photograph.'])
            return result

        step_started = time.perf_counter()
        cropped, crop_mask, _, crop_info = extract_retina(rgb, CONFIG)
        visual_steps.append({
            'title': 'Retinal crop',
            'description': 'Detected region of interest cropped before resizing.',
            'image': preview_data_url(cropped),
        })
        record_step(4, 'Retinal-mask extraction and crop',
                    'Detect the retina, form its mask, and crop to the detected bounding box.',
                    step_started, {
                        'mask coverage': round(crop_info['mask_fraction'], 4),
                        'mask fallback': crop_info['mask_fallback'],
                        'bounding box': [crop_info['bbox_x0'], crop_info['bbox_y0'],
                                         crop_info['bbox_x1'], crop_info['bbox_y1']],
                    })

        step_started = time.perf_counter()
        record_step(5, 'Crop geometry recorded',
                    'Record source dimensions, ROI bounds and crop dimensions for the audit log.',
                    step_started, {
                        'crop dimensions': f"{crop_info['cropped_width']} × {crop_info['cropped_height']}",
                        'source dimensions': f"{crop_info['original_width']} × {crop_info['original_height']}",
                    })

        step_started = time.perf_counter()
        prepared, mask, geometry = resize_pad(cropped, crop_mask, 224)
        visual_steps.append({
            'title': 'Resize and pad',
            'description': 'Aspect ratio preserved on a 224 × 224 canvas with even padding.',
            'image': preview_data_url(prepared),
        })
        record_step(6, 'Aspect-preserving resize',
                    'Resize the crop to the model canvas and pad without distorting its shape.',
                    step_started, {
                        'resized dimensions': f"{geometry['resized_height']} × {geometry['resized_width']}",
                        'output size': f"{geometry['target_size']} × {geometry['target_size']}",
                        'scale': round(geometry['scale_factor'], 5),
                        'padding (top, bottom, left, right)': [
                            geometry['padding_top'], geometry['padding_bottom'],
                            geometry['padding_left'], geometry['padding_right'],
                        ],
                        'interpolation': geometry['image_interpolation'],
                    })

        step_started = time.perf_counter()
        final = reconstruct_rgb(prepared, mask, CONFIG)
        visual_steps.append({
            'title': 'Model image',
            'description': 'RGB image after the inference mask is applied; this is the image representation supplied to the model.',
            'image': preview_data_url(final),
        })
        record_step(7, 'RGB reconstruction',
                    'Apply the selected inference preprocessing branch and mask the outside region.',
                    step_started, {
                        'selected branch': 'baseline RGB',
                        'outside-mask pixels': 'set to black',
                        'image shape': list(final.shape),
                        'pixel dtype': str(final.dtype),
                    })

        step_started = time.perf_counter()
        chw = np.ascontiguousarray(final.transpose(2, 0, 1))
        record_step(8, 'Channel layout',
                    'Convert the image from height-width-channel to channel-height-width.',
                    step_started, {'layout': 'CHW', 'shape': list(chw.shape), 'contiguous': chw.flags.c_contiguous})

        step_started = time.perf_counter()
        tensor = torch.from_numpy(chw).float().div(255)
        record_step(9, 'Pixel scaling',
                    'Convert 8-bit RGB pixel values to the [0, 1] range.',
                    step_started, {
                        'minimum': round(float(tensor.min()), 5),
                        'maximum': round(float(tensor.max()), 5),
                        'dtype': str(tensor.dtype),
                    })

        step_started = time.perf_counter()
        tensor = (tensor - self.mean) / self.std
        record_step(10, 'Backbone normalization',
                    'Normalize the input with the EfficientNet ImageNet channel statistics.',
                    step_started, {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})

        step_started = time.perf_counter()
        tensor = tensor.unsqueeze(0)
        if tensor.shape != (1, 3, 224, 224):
            raise ValueError(f'Unexpected model input shape: {list(tensor.shape)}.')
        record_step(11, 'Model input validation',
                    'Check the tensor shape and data type before inference.',
                    step_started, {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)})

        step_started = time.perf_counter()
        record_step(12, 'Quality-gate routing',
                    'Allow inference for accepted or reviewable images; a review flag remains visible.',
                    step_started, {
                        'quality decision': quality['quality_status'],
                        'classification permitted': quality['classification_permitted'],
                        'human review required': quality['human_review_required'],
                        'reason count': len(quality['reasons']),
                    }, status='review' if quality['human_review_required'] else 'complete')

        step_started = time.perf_counter()
        with LOCK, torch.inference_mode(), torch.random.fork_rng(devices=[]):
            self.model.eval()
            torch.manual_seed(self.routing['seed'])
            features = self.model.features(tensor)
            features = torch.flatten(self.model.avgpool(features), 1)
            record_step(13, 'EfficientNetB0 feature extraction',
                        'Run the normalized tensor through the trained backbone and global average pooling.',
                        step_started, {'feature shape': list(features.shape), 'checkpoint sha256': self.checkpoint_hash})
            step_started = time.perf_counter()
            self.model.classifier[0].train()
            try:
                logits = torch.stack([self.model.classifier(features)[0] for _ in range(self.routing['dropout_passes'])]).double()
            finally:
                self.model.eval()
            probabilities = (logits / self.routing['temperature']).softmax(dim=-1)
        record_step(14, 'Calibrated dropout inference',
                    'Sample classifier dropout and apply the saved calibration temperature.',
                    step_started, {
                        'dropout passes': self.routing['dropout_passes'],
                        'random seed': self.routing['seed'],
                        'temperature': self.routing['temperature'],
                        'calibration ready': self.routing['ready'],
                    })

        step_started = time.perf_counter()
        means = probabilities.mean(dim=0)
        grade = int(means.argmax())
        confidence = float(means[grade])
        uncertainty = float(probabilities[:, grade].std(correction=0))
        if not self.routing['ready']:
            result['reasons'].append('Routing calibration is insufficient; human review is required.')
        else:
            if confidence < self.routing['confidence_min']:
                result['reasons'].append('Prediction confidence is below the review threshold.')
            if uncertainty > self.routing['uncertainty_max']:
                result['reasons'].append('Dropout variability exceeds the review threshold.')
        result.update(status='review_required' if result['reasons'] else 'accepted',
                      classification_permitted=True, predicted_grade=grade, predicted_class=NAMES[grade],
                      confidence=confidence, uncertainty=uncertainty,
                      probabilities=[{'grade':i, 'name':name, 'probability':float(means[i])} for i,name in enumerate(NAMES)],
                      seconds=round(time.perf_counter()-started, 2))
        record_step(15, 'Prediction and review decision',
                    'Aggregate class probabilities, select the highest-scoring grade and apply review thresholds.',
                    step_started, {
                        'predicted grade': grade,
                        'predicted class': NAMES[grade],
                        'confidence': round(confidence, 5),
                        'dropout uncertainty': round(uncertainty, 5),
                        'confidence threshold': self.routing['confidence_min'],
                        'uncertainty threshold': self.routing['uncertainty_max'],
                        'final status': result['status'],
                    }, status='review' if result['status'] == 'review_required' else 'complete')
        return result
