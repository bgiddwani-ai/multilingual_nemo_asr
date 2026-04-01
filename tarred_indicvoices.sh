python /home/asr/NeMo/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
  --manifest_path='dataset/indicvoices/train/hi_manifest.json' \
  --target_dir='dataset/indicvoices/train/hi_tarred' \
  --num_shards=256 \
  --max_duration=20.0 \
  --min_duration=0.1 \
  --shuffle \
  --workers=16