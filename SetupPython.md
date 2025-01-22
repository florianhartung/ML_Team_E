# create the conda environment

<code>
conda create --name MAGICuda python=3.10
</code>

# install tools for MAGIC 

<code>
pip install matplotlib numpy pandas pyarrow ipython ipykernel scikit-learn 
</code>

# install torch on top conda with pip 

<code>
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
</code>
