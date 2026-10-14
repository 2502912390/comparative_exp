param(
    [string]$Python = "python",
    [string]$Root = "data\ogb",
    [string]$Output = "data\mag_paper_100k_sep_cls70_fill_pt"
)

& $Python tools\prepare_ogb_mag.py `
  --root $Root `
  --output $Output `
  --preprocess none `
  --paper_only `
  --max_papers 100000 `
  --max_classes 70 `
  --class_selection separability `
  --min_class_count 50 `
  --fill_to_max_papers `
  --target_edges 1831462 `
  --coauthor_edges_per_author 8 `
  --topic_edges_per_topic 16 `
  --seed 42

