
import chemprop

arguments = [
            '--data_path', '/home/abdulfatai/Downloads/Internship/GNN-Simulation/Rxncgr/dataset/full_rdb7/train.csv',
            '--separate_val_path', '/home/abdulfatai/Downloads/Internship/GNN-Simulation/Rxncgr/dataset/full_rdb7/val.csv',
            '--separate_test_path', '/home/abdulfatai/Downloads/Internship/GNN-Simulation/Rxncgr/dataset/full_rdb7/test.csv',
            '--dataset_type', 'regression',
            '--reaction',
            '--explicit_h',
            '--save_dir','chemprop_model',
            '--extra_metric', 'mae'
]
args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
