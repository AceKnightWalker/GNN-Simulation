declare -a configs=("ds.yaml" "dss.yaml")


# ZINC
for config in "${configs[@]}"
    do
    python Exp/run_experiment.py -grid "Configs/ZINC/${config}" -dataset "ZINC" --candidates 40  --repeats 10 --mode single
    done