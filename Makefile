get-convai2:
	python get_dataset.py --data_source convai2 --data_dir data/convai2

get-embeddings:
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
	gzip -d cc.en.300.vec.gz
	mkdir -p data/word2vec
	mv cc.en.300.vec data/word2vec

install-apex:
	git clone https://github.com/NVIDIA/apex
	cd apex
	pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	cd ../
