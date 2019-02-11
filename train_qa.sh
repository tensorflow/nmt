source="superuser.com"
data_dir=../qa_stackoverflow/seq2seq_data/$source
model_dir=models/model_qa_$source
mkdir -p $model_dir
python -m nmt.nmt \
    --src=doc --tgt=query \
    --vocab_prefix=$data_dir/vocab.text \
    --train_prefix=$data_dir/train.text \
    --dev_prefix=$data_dir/dev.text \
    --test_prefix=$data_dir/dev.text \
    --out_dir=$model_dir \
    --num_train_steps=120000 \
    --steps_per_stats=100 \
    --num_layers=4 \
    --num_units=200 \
    --batch_size=64 \
    --dropout=0.2 \
    --metrics=bleu \
    --src_max_len=300 \
    --residual=True
