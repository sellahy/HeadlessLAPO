Use Python 3.9.16o
Use cuda version 12.6.3
use gcc version 11.2.0

To install all the project dependencies, execute:
pip3 install -r all_reqs.txt

additionally, you must install fused operators from flash attention 2 as a dependency for TinyLlama. with your pip environment activated, execute:
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention

I modified headless-ad/src/tiny_llama/transformer.py to use PyTorch's implementation of flash attention, so there is no need to build/install flash attention itself.
