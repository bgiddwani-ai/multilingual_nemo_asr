python Speech/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
  --manifest_path='/data/dataset/indicvoices/train/mr_manifest.json' \
  --target_dir='/data/dataset/indicvoices/train/mr_tarred' \
  --num_shards=256 \
  --max_duration=30.0 \
  --min_duration=0.025 \
  --shuffle \
  --workers=16


python Speech/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
  --manifest_path='/data/dataset/indicvoices/train/gu_manifest.json' \
  --target_dir='/data/dataset/indicvoices/train/gu_tarred' \
  --num_shards=256 \
  --max_duration=30.0 \
  --min_duration=0.025 \
  --shuffle \
  --workers=16
