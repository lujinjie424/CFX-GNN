from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("public_run", ROOT / "run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)

sys.path.insert(0, str(ROOT / "cfx_gnn"))
main_spec = importlib.util.spec_from_file_location("cfx_main", ROOT / "cfx_gnn" / "main.py")
cfx_main = importlib.util.module_from_spec(main_spec); main_spec.loader.exec_module(cfx_main)


def test_all_configs_do_not_expose_seed_selection():
    for dataset in ("bail", "pokec_n", "pokec_z"):
        command = run.build_command(dataset, 0)
        assert "--seeds" not in command
        assert "--num_seeds" not in command
        assert "--force_retrain_disentangler" in command


def test_regular_runs_use_five_consecutive_seeds():
    assert cfx_main.experiment_seeds() == [1, 2, 3, 4, 5]
    assert cfx_main.experiment_seeds(smoke_test=True) == [1]


def test_smoke_test_reduces_all_stages():
    command = run.build_command("bail", 0, smoke_test=True)
    assert "--smoke_test" in command
    for key in ("pre_epochs", "epochs", "cf_warm_epochs", "cf_exp_epochs", "cf_joint_epochs"):
        assert command[command.index(f"--{key}") + 1] == "1"


def test_toy_config_builds_complete_pipeline_command():
    command = run.build_command("toy", 0)
    assert command[command.index("--dataset") + 1] == "toy"
    assert "--seeds" not in command
    assert command[command.index("--mask_feature_target_max") + 1] == "0.05"
    assert command[command.index("--mask_structure_target_max") + 1] == "0.05"


def test_method_variant_is_fixed_outside_dataset_configs():
    for dataset in ("bail", "pokec_n", "pokec_z", "toy"):
        command = run.build_command(dataset, 0)
        for key in run.PAPER_METHOD:
            assert f"--{key}" not in command


def test_missing_data_error_lists_required_caches():
    try:
        run.validate_data("pokec_z")
    except FileNotFoundError as error:
        message = str(error)
    else:
        raise AssertionError("validation unexpectedly accepted missing dataset files")
    assert "pokec_z_graph.bin" in message
    assert "pokec_z_index.bin" in message
