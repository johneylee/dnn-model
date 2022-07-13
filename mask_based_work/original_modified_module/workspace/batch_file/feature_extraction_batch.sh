python extract_features.py logmag_noisy ../data/speech/train feature_data/train_input
python extract_features.py logmag_noisy ../data/speech/devel feature_data/devel_input
python extract_features.py logmag_clean ../data/speech/train feature_data/train_ref_clean_logmag
python extract_features.py logmag_clean ../data/speech/devel feature_data/devel_ref_clean_logmag
python extract_features.py ibm ../data/speech/train feature_data/train_ref_ibm
python extract_features.py ibm ../data/speech/devel feature_data/devel_ref_ibm
python extract_features.py irm ../data/speech/train feature_data/train_ref_irm
python extract_features.py irm ../data/speech/devel feature_data/devel_ref_irm