python src/preprocess.py || exit 1
mkdir -p "results" || exit 1
python src/train.py || exit 1
