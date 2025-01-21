python infer_cv.py --num_classes 1 \
                    --feats_size 2048 \
                    --input_feats_path path/to/input/features \
                    --input_label_info_dict path/to/input/label_info_dict \
                    --input_model_path path/to/input/model.pth \
                    --output_path path/to/output/csv \
                    --average true \
                    --patch_max_loss_top_k 5