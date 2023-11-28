import gpt_2_simple as gpt2
import os
import requests

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "all_jokes.txt"

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              file_name,
              model_name=model_name,
              validate_every=100,
              experiment_name="finetune_upper", # Make sure to change if you re-run so it won't override.
              steps=1000,
              finetune_freeze_config=gpt2.finetune_all)

gpt2.generate(sess)

