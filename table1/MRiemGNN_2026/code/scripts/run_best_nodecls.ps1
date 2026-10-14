param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("taobao", "mag", "imdb", "alibaba")]
    [string]$Dataset,
    [string]$Python = "python",
    [string]$Device = "cuda"
)

switch ($Dataset) {
  "taobao" {
    & $Python main.py `
      --task node_cls `
      --mat_path data\small_alibaba_1_10\small_alibaba_1_10.mat `
      --split_mat_path data\small_alibaba_1_10\small_alibaba_1_10_split.mat `
      --resplit_labeled --train_ratio 0.9 --split_seed 1 `
      --feature_norm standard `
      --beta 0.8 --lambda_mutual 0.0 `
      --lr 0.01 --weight_decay 0.0005 --hid_dim 128 `
      --class_weight none `
      --node_relation_attention --node_relation_attention_mix 0.05 `
      --logit_prior_search --logit_prior_grid_min -0.2 --logit_prior_grid_max 0.2 --logit_prior_grid_steps 17 `
      --monitor_metric micro_f1 `
      --epochs 500 --patience 150 --log_every 999 `
      --device $Device --seed 1 --deterministic
  }
  "mag" {
    & $Python main.py `
      --task node_cls `
      --data_type pt --pt_dir data\mag_paper_100k_sep_cls70_fill_pt `
      --resplit_labeled --train_ratio 0.95 --split_seed 1 `
      --feature_norm none --no-composite_relations `
      --beta 0.8 --lambda_mutual 0.0 `
      --lr 0.005 --weight_decay 0.000005 --hid_dim 64 `
      --class_weight balanced --class_weight_strength 0.50 `
      --monitor_metric micro_f1 `
      --epochs 2000 --patience 350 --eval_every 100 --log_every 500 `
      --device $Device --seed 2 --deterministic
  }
  "imdb" {
    & $Python main.py `
      --task node_cls `
      --data_type pt --pt_dir data\imdb_sr_rsc_pt `
      --resplit_labeled --train_ratio 0.95 --split_seed 1 `
      --feature_norm none --no-add_reverse_edges `
      --beta 0.75 --lambda_mutual 0.0 `
      --lr 0.0065 --weight_decay 0.000005 `
      --hid_dim 64 --classifier_hidden 32 `
      --class_weight none --focal_gamma 1.35 `
      --logit_prior_search --logit_prior_grid_min -1.0 --logit_prior_grid_max 0.4 --logit_prior_grid_steps 29 `
      --monitor_metric micro_f1 `
      --epochs 500 --patience 180 --log_every 999 `
      --device $Device --seed 1 --deterministic
  }
  "alibaba" {
    & $Python main.py `
      --task node_cls `
      --mat_path external\BPHGNN\data\alibaba_small\alibaba_small.mat `
      --split_mat_path external\BPHGNN\data\alibaba_small\alibaba_small_20.mat `
      --feature_path external\BPHGNN\data\alibaba_small\alibaba_small_encode.pt `
      --resplit_labeled --train_ratio 0.9 --split_seed 3 `
      --feature_norm none `
      --beta 0.8 --lambda_mutual 0.0 `
      --lr 0.012 --weight_decay 0.000005 --hid_dim 128 `
      --class_weight balanced --class_weight_strength 0.2 `
      --logit_prior_alpha 1.2 `
      --monitor_metric micro_f1 `
      --epochs 1400 --patience 450 --log_every 999 `
      --device $Device --seed 3 --deterministic
  }
}

