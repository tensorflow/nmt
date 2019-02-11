source="superuser.com"
data_dir=../qa_stackoverflow/seq2seq_data/$source
model_dir=models/model_qa_$source
grpc_port=8500
port=$(($grpc_port+1))
service_name=serving_$port
pwd=$(pwd)
echo "service_name=$service_name"

docker kill $service_name
docker rm $service_name
rm -r $model_dir/154*
python -m nmt.nmt \
    --out_dir=$model_dir \
    --export_path=$model_dir \
    --ckpt_path=$model_dir \
    --model=discriminative \
#    --infer_mode="beam_search" \
#    --beam_width=10
docker run --name $service_name -p $port:8501 --mount type=bind,source=$pwd/$model_dir,target=/models/seq2seq -e MODEL_NAME=seq2seq -t tensorflow/serving

# enables GPU serving
# docker run --name $service_name --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p $port:8501 --mount type=bind,source=$pwd/$model_dir,target=/models/seq2seq -e MODEL_NAME=seq2seq -t tensorflow/serving:latest-gpu
