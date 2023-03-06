# Soxan


This repository consists of models, scripts, and notebooks that help you to use all the benefits of Wav2Vec 2.0 in your
research. In the following, I'll show you how to train speech tasks in your dataset and how to use the pretrained
models.

I have modified the source code with the following changes to get this to work on my Mac Pro local env
Diff from source code:
(1) Extra files - data_prep.py and inference.py, (2) a modification to the bash script for training,
(3) python3 shebang included in run_wav2vec.py 

## How to train

I'm just at the beginning of all the possible speech tasks. To start, we continue the training script with the speech
emotion recognition problem.

### Training - Notebook

| Task                                     | Notebook                                                                                                                                                                                                            |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Speech Emotion Recognition (Wav2Vec 2.0) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb) |
| Speech Emotion Recognition (Hubert)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_HuBERT.ipynb)   |
| Audio Classification (Wav2Vec 2.0)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb)             |

### Training - CMD

```bash
python3 run_wav2vec_clf.py \
    --pooling_mode="mean" \
    --model_name_or_path="lighteternal/wav2vec2-large-xlsr-53-greek" \
    --model_mode="wav2vec2" \ # or you can use hubert
    --output_dir=/path/to/output \
    --cache_dir=/path/to/cache/ \
    --train_file=/path/to/train.csv \
    --validation_file=/path/to/dev.csv \
    --test_file=/path/to/test.csv \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --num_train_epochs=5.0 \
    --evaluation_strategy="steps"\
    --save_steps=100 \
    --eval_steps=100 \
    --logging_steps=100 \
    --save_total_limit=2 \
    --do_eval \
    --do_train \
    --fp16=False \ #had set to false since the library dependency issue could not be resolved
    --freeze_feature_extractor
```

### Prediction
#use the inference.py file in src if this doesn't work
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification

model_name_or_path = "path/to/your-pretrained-model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate

# for wav2vec
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

# for hubert
model = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs


path = "/path/to/disgust.wav"
outputs = predict(path, sampling_rate)    
```

Output:

```bash
[
    {'Emotion': 'anger', 'Score': '0.0%'},
    {'Emotion': 'disgust', 'Score': '99.2%'},
    {'Emotion': 'fear', 'Score': '0.1%'},
    {'Emotion': 'happiness', 'Score': '0.3%'},
    {'Emotion': 'sadness', 'Score': '0.5%'}
]
```


## Demos

| Demo                                                     | Link                                                                                                               |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Speech To Text With Emotion Recognition (Persian) - soon | [huggingface.co/spaces/m3hrdadfi/speech-text-emotion](https://huggingface.co/spaces/m3hrdadfi/speech-text-emotion) |


## Models

| Dataset                                                                                                                      | Model                                                                                                                                           |
|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| [ShEMO: a large-scale validated database for Persian speech emotion detection](https://github.com/mansourehk/ShEMO)          | [m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition) |
| [ShEMO: a large-scale validated database for Persian speech emotion detection](https://github.com/mansourehk/ShEMO)          | [m3hrdadfi/hubert-base-persian-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/hubert-base-persian-speech-emotion-recognition)     |
| [ShEMO: a large-scale validated database for Persian speech emotion detection](https://github.com/mansourehk/ShEMO)          | [m3hrdadfi/hubert-base-persian-speech-gender-recognition](https://huggingface.co/m3hrdadfi/hubert-base-persian-speech-gender-recognition)       |
| [Speech Emotion Recognition (Greek) (AESDD)](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)              | [m3hrdadfi/hubert-large-greek-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/hubert-large-greek-speech-emotion-recognition)       |
| [Speech Emotion Recognition (Greek) (AESDD)](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)              | [m3hrdadfi/hubert-base-greek-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/hubert-base-greek-speech-emotion-recognition)         |
| [Speech Emotion Recognition (Greek) (AESDD)](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)              | [m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition)     |
| [Eating Sound Collection](https://www.kaggle.com/mashijie/eating-sound-collection)                                           | [m3hrdadfi/wav2vec2-base-100k-eating-sound-collection](https://huggingface.co/m3hrdadfi/wav2vec2-base-100k-eating-sound-collection)             |
| [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) | [m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres](https://huggingface.co/m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres)                       |


## Verified Training Log
```bash
FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
***** Running training *****
  Num examples = 422
  Num Epochs = 5
  Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 8
  Gradient Accumulation steps = 2
  Total optimization steps = 265
  Number of trainable parameters = 312283269
{'loss': 1.259, 'learning_rate': 6.226415094339622e-05, 'epoch': 1.89}
 38%|██████████████████████████████████████████████▊                                                                             | 100/265 [9:41:41<1:39:22, 36.14s/it]The following columns in the evaluation set don't have a corresponding argument in `Wav2Vec2ForSpeechClassification.forward` and have been ignored: emotion, path, name. If emotion, path, name are not expected by `Wav2Vec2ForSpeechClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 91
  Batch size = 4
{'eval_loss': 0.999789297580719, 'eval_accuracy': 0.5824176073074341, 'eval_runtime': 111.5244, 'eval_samples_per_second': 0.816, 'eval_steps_per_second': 0.206, 'epoch': 1.89}
 38%|██████████████████████████████████████████████▊                                                                             | 100/265 [9:43:32<1:39:22, 36.14s/it]
Saving model checkpoint to /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/checkpoint-100
Configuration saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/checkpoint-100/config.json
Model weights saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/checkpoint-100/pytorch_model.bin
Feature extractor saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/checkpoint-100/preprocessor_config.json
{'loss': 0.4857, 'learning_rate': 2.4528301886792453e-05, 'epoch': 3.77}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████▎                              | 200/265 [10:38:36<31:10, 28.78s/it]The following columns in the evaluation set don't have a corresponding argument in `Wav2Vec2ForSpeechClassification.forward` and have been ignored: emotion, path, name. If emotion, path, name are not expected by `Wav2Vec2ForSpeechClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 91
  Batch size = 4
{'eval_loss': 0.6223454475402832, 'eval_accuracy': 0.8571428656578064, 'eval_runtime': 107.8336, 'eval_samples_per_second': 0.844, 'eval_steps_per_second': 0.213, 'epoch': 3.77}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████▎                              | 200/265 [10:40:24<31:10, 28.78s/it]
Saving model checkpoint to /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/checkpoint-200
Configuration saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/checkpoint-200/config.json
Model weights saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/checkpoint-200/pytorch_model.bin
Feature extractor saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/checkpoint-200/preprocessor_config.json
 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████▊                     | 220/265 [10:50:41<22:22, 29.82s/it]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 265/265 [11:16:47<00:00, 29.74s/it]
Training completed. Do not forget to share your model on huggingface.co/models =)
{'train_runtime': 40607.4862, 'train_samples_per_second': 0.052, 'train_steps_per_second': 0.007, 'train_loss': 0.7086862851988595, 'epoch': 5.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 265/265 [11:16:47<00:00, 153.24s/it]
Saving model checkpoint to /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/
Configuration saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/config.json
Model weights saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/pytorch_model.bin
Feature extractor saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/preprocessor_config.json
Feature extractor saved in /Users/ashwic/Desktop/CSEP590/SER/w2v2ser/soxan/model_files/output/preprocessor_config.json
***** train metrics *****
  epoch                    =         5.0
  train_loss               =      0.7087
  train_runtime            = 11:16:47.48
  train_samples            =         422
  train_samples_per_second =       0.052
  train_steps_per_second   =       0.007
02/27/2023 11:19:09 - INFO - __main__ -   *** Evaluate ***
The following columns in the evaluation set don't have a corresponding argument in `Wav2Vec2ForSpeechClassification.forward` and have been ignored: emotion, path, name. If emotion, path, name are not expected by `Wav2Vec2ForSpeechClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 91
  Batch size = 4
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [01:58<00:00,  5.17s/it]
***** eval metrics *****
  epoch                   =        5.0
  eval_accuracy           =     0.8352
  eval_loss               =      0.629
  eval_runtime            = 0:02:04.11
  eval_samples            =         91
  eval_samples_per_second =      0.733
  eval_steps_per_second   =      0.185
```
