rlaunch --cpu=2 --gpu=1 --memory=20000 --preemptible=no --max-wait-time 1000s --negative-tags=titanx \
 -- python3 debug.py \
 --dataset market1501 \
 --data_dir '/unsullied/sharefs/hanchuchu/isilon-home/' \
 --logs_dir '/unsullied/sharefs/hanchuchu/isilon-home/train_log/domain' \