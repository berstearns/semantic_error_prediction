echo "featurising..."
l
wget https://osf.io/download/6kauf/ -O data/AoA_51715_words.xlsx

libreoffice --headless --convert-to tsv data/AoA_51715_words.xlsx --outdir data

python featurise_bea.py data/train/train.tsv data/AoA_51715_words.tsv data/vocab_lists/cefr_list.tsv
python featurise_bea.py data/train/new_dev.tsv data/AoA_51715_words.tsv data/vocab_lists/cefr_list.tsv
python featurise_bea.py data/dev/all_dev.tsv data/AoA_51715_words.tsv data/vocab_lists/cefr_list.tsv
