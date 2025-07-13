declare -a configs=("gcn_cre.yaml" "gcn_sbe_ds.yaml" "gcn_sbe_dss.yaml" "gcn.yaml")



# ZINC
for config in "${configs[@]}"
    do
    python Exp/run_experiment.py -grid "Configs/GCN/ZINC/${config}" -dataset "ZINC" --candidates 40  --repeats 10 --mode single
    done