python3 train_cv_fold_nni.py --num_classes 1 \
                    --feats_size 2048 \
                    --dropout_patch 0.95 \
                    --input_feats_path path/to/input/features \
                    --input_label_info_dict path/to/input/label_info_dict \
                    --output_dir path/to/output/dir \
                    --num_try 1 \
                    --average false \
                    --random_dropout_patch true \
                    --cosine_cycle_steps 25 \
                    --cosine_cycle_warmup 2 \
                    --num_epochs 100 \
                    --min_lr 1e-4 \
                    --lr 1e-3 \
                    --bag_loss_weight 0.7 \
                    --patch_max_loss_weight 0.3 \
                    --patch_max_loss_top_k 20
                    