pdm run python litfass/generate.py \
--hub "cdminix/litfass_a" \
--sentence "I can speak a bit faster too when it's needed." \
--output_path . \
--use_voicefixer True \
--speaker ../data/train-clean-a/5570 \
--prior_values 1 0 100 -1