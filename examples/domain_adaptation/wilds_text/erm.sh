# civilcomments
# group by ['y', 'black'] and reweight
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "civilcomments" --lr 1e-05  --opt-level O1 --deterministic \
    --log logs/erm/civilcomments --unlabeled-list "extra_unlabeled" --metric "acc_wg" --seed 0 \
    --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black

# amazon
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon -b 24 24 \
    --metric "10th_percentile_acc" --lr 1e-5 --max_token_length 512 --wd 0.01 --seed 1 --epochs 3 --deterministic