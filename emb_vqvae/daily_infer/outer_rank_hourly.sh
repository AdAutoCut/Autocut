# unzip -d /home/ /share/ad/baixuehan03/hadoop.zip
# unzip -d /root/anaconda3/envs/ /data/phd/SPU/spu_pretrain/pytorch2.0.zip
unset -v http_proxy https_proxy no_proxy
export HADOOP_USER_NAME=ad_algo_g
# export HIVE_GROUP_ID=572
export LANG=en_US.UTF-8
export LANG=zh_CN.UTF-8

p_date=$(date -d "2 hour ago" "+%Y%m%d")
p_hourmin=$(date -d "2 hour ago" +"%H")00
echo ${p_date}
echo ${p_hourmin}

photo_emb_hdfs_path=/home/ad_algo_g/dw/ad_algo_g.db/mmu_ia_embedding_hourly_textfile/p_date=${p_date}/p_hourmin=${p_hourmin}
code_path=/phd/content_ID/online_server/outer_loop/outer_recall/emb_vqvae


daily_path_output=/phd/content_ID/online_server/outer_loop/outer_recall/photo_sid/hive_files/${p_date}/${p_hourmin}
if [ ! -d ${daily_path_output} ]; then
    mkdir -p ${daily_path_output}
fi



while :
do
    /home/hadoop/software/hadoop/bin/hadoop fs -test -e ${photo_emb_hdfs_path}
    if [ $? -ne 1 ]; then
      echo "Directory exists! ${photo_emb_hdfs_path}"
      break
    else
      echo "Directory not exists! ${photo_emb_hdfs_path}"
	    sleep 1m
    fi
done



# # python ${code_path}/dask_get_hdfs_data.py --hdfs_path ${photo_emb_hdfs_path} --local_path ${daily_path_output}

# step1：拉取 hive 文件
/home/hadoop/software/hadoop/bin/hadoop fs -getmerge ${photo_emb_hdfs_path}  ${daily_path_output}/photo_emb.txt




 # step2: 获取 photo 对应的 sid
cd ${code_path}

#CUDA_VISIBLE_DEVICES=0 /data/jiajian/conda_env/qwen/bin/python -m torch.distributed.launch  --nproc_per_node=8 \
CUDA_VISIBLE_DEVICES=0 /data/jiajian/conda_env/qwen/bin/torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
        --master_addr='localhost' --master_port=10011 ${code_path}/daily_infer/online_infer.py \
        --config-file ./configs/daily_outer.yaml \
        --infer-file ${daily_path_output}/photo_emb.txt \
        --output-file ${daily_path_output}/photo_sid.txt \
        --data-type 2



#CUDA_VISIBLE_DEVICES=3 /data/jiajian/conda_env/qwen/bin/torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
#        --master_addr='localhost' --master_port=10011 /phd/content_ID/online_server/outer_loop/outer_recall/emb_vqvae/daily_infer/online_infer.py \
#        --config-file ./configs/daily_outer.yaml \
#        --infer-file /phd/content_ID/online_server/outer_loop/outer_recall/photo_sid/hive_files/history/all.txt \
#        --output-file /phd/content_ID/online_server/outer_loop/outer_recall/photo_sid/hive_files/history/photo_sid.txt \
#        --data-type 2




while :
do
    if [ -s ${daily_path_output}/photo_sid.txt ]; then
        echo "${daily_path_output}/photo_sid.txt file is not null"
        break
    else
        echo "${daily_path_output}/photo_sid.txt file is null"
        sleep 1m
    fi
done



# /home/hadoop/software/hadoop/bin/hadoop fs -mkdir -p /home/ad_algo_g/users/jiajian/content_id/outer_rank/

/home/hadoop/software/hadoop/bin/hadoop fs -put -f ${daily_path_output}/photo_sid.txt /home/ad_algo_g/users/jiajian/content_id/outer_rank/photo_sid_${p_date}_${p_hourmin}.txt
/home/hadoop/software/hive/bin/hive -e "set hive.support.concurrency=false; load data inpath '/home/ad_algo_g/users/jiajian/content_id/outer_rank/photo_sid_${p_date}_${p_hourmin}.txt' overwrite into table ad_algo_g.outer_recall_sid_h partition (p_date='${p_date}', p_hourmin='${p_hourmin}')"

# 删除前两天的数据
# 获取两天前的日期，格式为 YYYY-MM-DD
TARGET_DATE=$(date -d "4 days ago" +%Y%m%d)

# 设置文件夹路径前缀
TARGET_FOLDER=/phd/content_ID/online_server/outer_loop/outer_recall/photo_sid/hive_files/${TARGET_DATE}



# # 检查文件夹是否存在
if [ -d "$TARGET_FOLDER" ]; then
  echo "Folder $TARGET_FOLDER exists. Deleting..."
  rm -rf ${TARGET_FOLDER}
  echo "Folder $TARGET_FOLDER deleted."
else
  echo "Folder $TARGET_FOLDER does not exist."
fi

echo "Done!"
