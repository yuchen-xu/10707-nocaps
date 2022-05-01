# Code to run baseline
python scripts/inference.py \
    --config /path/to/config.yaml \
    --checkpoint-path /path/to/checkpoint.pth \
    --output-path /path/to/save/predictions.json \

python baseline/scripts/train.py \
    --config baseline/configs/updown_nocaps_val.yaml \
    --config-override OPTIM.BATCH_SIZE 32 \
    --gpu-ids -1 --serialization-dir baseline/checkpoints/updown-baseline

python baseline/scripts/inference.py \
    --config baseline/configs/updown_nocaps_test.yaml \
    --checkpoint-path baseline/checkpoints/updown-baseline/checkpoint_10000.pth \
    --output-path predictions.json \

# Code to run classifier
python classifier/train_sim.py \
    --config configs/updown_nocaps_val.yaml
    --output_dir checkpoints/sim

python classifier/train_natural.py \
    --config configs/updown_nocaps_val.yaml
    --output_dir checkpoints/sim

python predict_sim.py