$conda = 'C:\Users\wangx\miniconda3\Scripts\conda.exe'
# 使用同一套配置，只改变随机种子，观察结果稳定性。
$seeds = @(41, 42, 43)
foreach ($seed in $seeds) {
    & $conda run -n py310 python scripts/train_dssm.py --epochs 2 --max-users 1000 --batch-size 512 --seed $seed --negative-mode in_batch
}
