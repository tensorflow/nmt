docker run --name serving -p 8501:8501 --mount type=bind,source=/Users/zyin/Desktop/server/zi/nmt/models/model_qa_superuser.com,target=/models/seq2seq -e MODEL_NAME=seq2seq -t tensorflow/serving
# for serving the generative model
curl -d '{"inputs": ["is it possible to use an ssd as a large capacity flash drive i know that some computer cases include external sata ports so would it be reasonable to assume that one could interface with the ssd using a sata cable"]}' -X POST http://localhost:8501/v1/models/seq2seq:predict -s -w 'Total: %{time_total}\n'
# for serving the scoring (likelihood) model
curl -d '{"inputs": {"seq_input_src": ["is it possible to use an ssd as a large capacity flash drive i know that some computer cases include external sata ports so would it be reasonable to assume that one could interface with the ssd using a sata cable"],"seq_input_tgt": ["use ssd as external storage"]}}' -X POST http://localhost:8501/v1/models/seq2seq:predict -s -w 'Total: %{time_total}\n'
# for syncing data and model
rsync -r --exclude='*/*' ./nmt/ zi@archimedes.elca.mw.int:~/nmt/nmt/
