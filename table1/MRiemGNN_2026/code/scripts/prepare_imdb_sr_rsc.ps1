param(
    [string]$Python = "python",
    [string]$InputDir = "external\SR-RSC\dataset\imdb",
    [string]$Output = "data\imdb_sr_rsc_pt"
)

& $Python tools\prepare_sr_rsc.py `
  --input_dir $InputDir `
  --output $Output

