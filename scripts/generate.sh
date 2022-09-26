pdm run python litfass/generate.py \
--hub "cdminix/litfass_a" \
--sentence "ALMOST AT THE SAME TIME D'ALBRET ARRIVED IN QUEST OF HIS CARDINAL'S HAT." \
--output_path . \
--use_voicefixer True \
--speaker ../data/train-clean-a/4013 \
--prior_values -1 -1 -1 -1