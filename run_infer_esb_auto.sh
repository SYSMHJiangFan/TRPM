python infer_cv_esb_auto_rec.py --num_classes 1 \
                    --feats_size 2048 \
                    --input_feats_path path/to/input/features \
                    --input_label_info_dict path/to/input/label_info_dict \
                    --input_model_exp_dir path/to/input/model_exp_dir \
                    --postfix postfix_value \
                    --output_dir path/to/output/dir \
                    --average true \
                    --patch_max_loss_top_k 1