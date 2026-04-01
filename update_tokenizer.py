import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
asr_model = nemo_asr.models.ASRModel.restore_from("models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo")

new_tokenizer_cfg = OmegaConf.create({'type': 'agg', 'langs': {}})
english_tokenizer_cfg = OmegaConf.create({'dir':  'models/en_tokenizer_tdt0_6b_v2/tokenizer_spe_bpe_v1024', 'type': 'bpe'})
hindi_tokenizer_cfg = OmegaConf.create({'dir': 'models/hi_tokenizer_v256/tokenizer_spe_bpe_v256', 'type': 'bpe'})
new_tokenizer_cfg.langs['en'] = english_tokenizer_cfg
new_tokenizer_cfg.langs['hi'] = hindi_tokenizer_cfg

asr_model.change_vocabulary(
        new_tokenizer_dir=new_tokenizer_cfg,
        new_tokenizer_type="agg",
    )

asr_model.save_to("models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_en_hi_init.nemo")