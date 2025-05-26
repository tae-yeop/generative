# 일부만 받기
git lfs install
git clone --depth 1 https://huggingface.co/h94/IP-Adapter
cd IP-Adapter
git lfs pull -I "config.json, pytorch_model.bin"

# 일부만 받기2
# ?download=true 붙히기 + resolve로 바꾸기
wget --content-disposition "https://huggingface.co/Doubiiu/ToonCrafter/resolve/main/model.ckpt?download=true"
wget --content-disposition "https://huggingface.co/Doubiiu/ToonCrafter/resolve/main/sketch_encoder.ckpt?download=true"

# 전체 다운받기 1
git clone https://huggingface.co/h94/IP-Adapter

# 전체 다운받기 2
huggingface-cli lfs-enable-largefiles .
mkdir models
git clone https://huggingface.co/dreMaz/AnimeInstanceSegmentation models/AnimeInstanceSegmentation