# 生成图片(英文、汉字、数字)
python bin/deep_ocr_make_caffe_dataset
--out_caffe_dir workspace/caffe_dataset_all
--font_dir fonts/chinese_fonts
--width 28 --height 28 --margin 0
--langs eng+chi+digits

# 生成数据库
cd data/caffe_nets/all
./create_lmdb.sh


# 训练模型
./train_lmdb.sh

# 测试识别
python test_all.py

# 生成图片(数字)
python deep_ocr_make_caffe_dataset  \
--out_caffe_dir workspace/caffe_dataset_digits  \
--font_dir fonts/chinese_fonts \
--width 28 --height 28 --margin 0   \
--langs digits

# 生成数据库
cd data/caffe_nets/digits
./create_lmdb.sh




