## project

解决仓储环境中货架移动以及回归问题

### r_para
指的是货架回归位置包括了初始公共的回归位以及其他货架离开后空出的位置


To train the attention models for r_para
```bash
python run.py --problem r_para --batch_size 256 --epoch_size 12800 --lr_model 0.0001 --n_epochs 100 --eval_batch_size 1000 --checkpoint_epochs 10 --ss_size 10 --r_size 10  --run_name train
```

To validate the trained_model:
```bash
python run.py --eval_only --problem r_para --eval_batch_size 1000 --ss_size 10 --r_size 10  --run_name validate --val_dataset '#path' --load_path '#model_path'
```



The code are developed based on :
https://github.com/wouterkool/attention-learn-to-route