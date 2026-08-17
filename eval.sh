CUDA_VISIBLE_DEVICES=0 python eval_manifest.py \
    --model=/home/multilingual_nemo_asr/results/experiments_5/hi_mr_gu_nemotron_pretraining/checkpoints/hi_mr_gu_nemotron_pretraining.nemo \
    --manifest="/sadata/speech/asr-data/manifests/hi_valid_v3.jsonl" \
    --target-lang=hi-IN \
    --output="results/experiments_5/hi_valid_v3_preds.jsonl" \
    --metrics-json="results/experiments_5/hi_valid_v3_metrics.jsonl" \
    --batch-size=16 
