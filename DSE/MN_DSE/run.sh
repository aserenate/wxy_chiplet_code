python simplified_multi_network_DSE_re.py \
	--architecture simba \
	--nn_list_type nlp \
	--chiplet_num 16 \
	--mem_num 4 \
	--Optimization_Objective latency \
	--BW_Reallocator_tag 0 \
	--layout_mapping_method random \
	--tp_TH 0.2 \
	--sp_TH 1 \
	--topology mesh \
	--partition_ratio 0.4

# --nn_list resnet50+darknet19+vit+Unet \
# --nn_list GNMT+BERT+ncf \
# --nn_list resnet18+VGG16+GNMT+BERT \