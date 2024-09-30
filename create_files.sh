echo "extracting m2..."

wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz -P data

tar -xvzf data/wi+locness_v2.1.bea19.tar.gz -C data/

mkdir data/wi+locness/long_m2
mkdir data/wi+locness/inverted_long_m2
mkdir data/train
mkdir data/dev


for filename in data/wi+locness/json/*.json; do
    python json_to_m2.py "$filename" -gold -out "data/wi+locness/long_m2/$(basename "$filename" .json)" &
done

wait

echo "Moving to inversion..."

for filename in data/wi+locness/long_m2/*.m2; do
    python invert_m2.py "$filename" -out "data/wi+locness/inverted_long_m2/$(basename "$filename")"
done

echo "Convert m2 to tsv (dev)..."

for filename in data/wi+locness/inverted_long_m2/*.dev.m2; do
    python m2_conversion.py "$filename" "data/wi+locness/long_m2/$(basename "$filename" .m2).cefr.json" "data/dev/$(basename "$filename" .m2).tsv" &
done

echo "Convert m2 to tsv (train)..."

for filename in data/wi+locness/inverted_long_m2/*.train.m2; do
    python m2_conversion.py "$filename" "data/wi+locness/long_m2/$(basename "$filename" .m2).cefr.json" "data/train/$(basename "$filename" .m2).tsv" &
done

wait

echo "Combining tsv (train)..."

python combine_tsv.py data/train/*.train.tsv data/train/all_train.tsv

echo "Combining tsv (dev)..."

python combine_tsv.py data/dev/*.dev.tsv data/dev/all_dev.tsv

echo "Creating new train-dev split..."

python split_tsv.py data/train/all_train.tsv data/train/ --create_dev False

mv data/train/test.tsv data/train/new_dev.tsv
