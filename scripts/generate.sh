pdm run python litfass/generate.py \
--hub "cdminix/litfass_a" \
--sentence '"you are insane!", he screamed in anguish.' \
--output_path . \
--use_voicefixer True \
--speaker ../data/train-clean-a/3922 \
--prior_values -1 -1 -1 -1