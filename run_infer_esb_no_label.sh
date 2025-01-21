
python infer_cv_esb_no_label.py --num_classes 1 \
                    --feats_size 2048 \
                    --input_feats_path path/to/input/features \
                    --input_label_info_dict path/to/input/label_info_dict \
                    --input_model_rec_path path/to/input/model_rec_path \
                    --output_path path/to/output/csv \
                    --average true \
                    --patch_max_loss_top_k 1
