start=$(date +%s)

python3 -m pyserini.encode input  --corpus corpus/msmarco-passage-mini/corpus --fields text title  --docid-field docid output --embeddings corpus/msmarco-passage-mini/pyserini  --to-faiss encoder  --encoder castorini/mdpr-tied-pft-msmarco --encoder-class auto --batch-size 8 --max-length 128 --device cpu

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
