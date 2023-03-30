# 消融实验
# train_noisy_60_noweight
python train_without_weight.py --train_csv "data/csv/train_noisy_60.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_noisy_60_noweight --epochs 100 --batch-size 4 --learning-rate 0.1 

# train_noisy_60_miouweight_noKD
python train_with_Teacher_weight.py --train_csv "data/csv/train_noisy_60_PointRend_weight_miouscore.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_noisy_60_miouweight_noKD --epochs 100 --batch-size 4 --learning-rate 0.1

# train_noisy_60_bfweight_noKD
python train_with_Teacher_weight.py --train_csv "data/csv/train_noisy_60_PointRend_weight_bfscore.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_noisy_60_bfweight_noKD --epochs 100 --batch-size 4 --learning-rate 0.1

# train_noisy_60_bmweight_noKD
python train_with_Teacher_weight.py --train_csv "data/csv/train_noisy_60_PointRend_weight_bmscore.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_noisy_60_bmweight_noKD --epochs 100 --batch-size 4 --learning-rate 0.1

# train_noisy_60_miouweight_KD
python train_with_KDweight.py --train_csv "data/csv/train_noisy_60_PointRend_weight_miouscore.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_noisy_60_miouweight_KD --epochs 100 --batch-size 4 --learning-rate 0.1

# train_noisy_60_bfweight_KD
python train_with_KDweight.py --train_csv "data/csv/train_noisy_60_PointRend_weight_bfscore.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_noisy_60_bfweight_KD --epochs 100 --batch-size 4 --learning-rate 0.1

# train_noisy_60_bmweight_KD
python train_with_KDweight.py --train_csv "data/csv/train_noisy_60_PointRend_weight_bmscore.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_noisy_60_bmweight_KD --epochs 100 --batch-size 4 --learning-rate 0.1


# 与其他方法对比
# train_noisy_60_decouping
python train_decoupling.py --train_csv "data/csv/train_noisy_60.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_decoupling --epochs 100 --batch-size 4 --learning-rate 0.1

# train_noisy_60_co_teaching
python train_co_teaching.py --train_csv "data/csv/train_noisy_60.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_co_teaching --epochs 100 --batch-size 4 --learning-rate 0.1

# train_noisy_60_co_teaching+
python train_co_teaching+.py --train_csv "data/csv/train_noisy_60.csv" --valid_csv "data/csv/valid_noisy_60.csv" --test_csv "data/csv/test.csv" --name train_co_teaching+ --epochs 100 --batch-size 4 --learning-rate 0.1