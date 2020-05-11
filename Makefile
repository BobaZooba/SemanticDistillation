get-embeddings:
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
	gzip -d cc.en.300.vec.gz
	mkdir -p data
	mv cc.en.300.vec data/

install-apex:
	git clone https://github.com/NVIDIA/apex
	cd apex
	pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	cd ../
