CUDA_VISIBLE_DEVICES=5 python src/iterative_cluster.py \
  --data_path processed_data/arxiv_label_3.5k \
  --exp_dir experiments/arxiv_label \
  --proposer_model gpt-4o-mini \
  --assigner_name google/gemma-2-2b-it \
  --proposer_num_descriptions_to_propose 10 \
  --assigner_for_proposed_descriptions_template templates/gemma_assigner.txt \
  --cluster_num_clusters 7 \
  --min_cluster_fraction 0.0 \
  --max_cluster_fraction 0.3 \
  --cluster_overlap_penalty 0.6\
  --with_label \
  --verbose