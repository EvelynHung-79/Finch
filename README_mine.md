python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

nohup bash run_experiment.sh > experiment.log 2>&1 &
nohup bash ./scripts/longbench/run_everything_compress_llama.sh > run.log 2>&1 &