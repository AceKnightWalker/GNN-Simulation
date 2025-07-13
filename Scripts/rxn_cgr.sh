python Exp/run_experiment.py -grid Configs/RXNCGR/dss.yaml -dataset rxn_cgr --candidates 6 --repeats 5 --mode single
python Exp/run_experiment.py -grid Configs/RXNCGR/gin.yaml -dataset rxn_cgr --candidates 12 --repeats 2 --mode single
python Exp/run_experiment.py -grid Configs/RXNCGR/cre.yaml -dataset rxn_cgr --candidates 12 --repeats 2 --mode single
python Exp/run_experiment.py -grid Configs/RXNCGR/sbe_dss.yaml -dataset rxn_cgr --candidates 12 --repeats 2  --mode single
python Exp/run_experiment.py -grid Configs/RXNCGR/ds.yaml -dataset rxn_cgr --candidates 6  --repeats 2 --mode single
python Exp/run_experiment.py -grid Configs/RXNCGR/sbe_ds.yaml -dataset rxn_cgr --candidates 12 --repeats 2 --mode single
python Exp/run_experiment.py -grid Configs/MLP/RXNCGR/mlp.yaml -dataset rxn_cgr --candidates 12 --repeats 2 --mode single
python Exp/run_experiment.py -grid Configs/MLP/RXNCGR/mlp_cre.yaml -dataset rxn_cgr --candidates 12  --repeats 2 --mode single
python Exp/run_experiment.py -grid Configs/MLP/RXNCGR/mlp_sbe_dss.yaml -dataset rxn_cgr --candidates 12 --repeats 2 --mode single
python Exp/run_experiment.py -grid Configs/MLP/RXNCGR/mlp_sbe_ds.yaml -dataset rxn_cgr --candidates 12 --repeats 2 --mode single


