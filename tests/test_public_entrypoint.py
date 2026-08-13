from pathlib import Path
import importlib.util


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("public_run", ROOT / "run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


def test_all_configs_build_five_seed_commands():
    for dataset in ("bail", "pokec_n", "pokec_z"):
        command = run.build_command(dataset, 0, None)
        assert command[command.index("--seeds") + 1] == "1,2,3,10,11"
        assert command[command.index("--num_seeds") + 1] == "5"
        assert "--force_retrain_disentangler" in command


def test_smoke_test_reduces_all_stages():
    command = run.build_command("bail", 0, None, smoke_test=True)
    assert command[command.index("--seeds") + 1] == "1"
    for key in ("pre_epochs", "epochs", "cf_warm_epochs", "cf_exp_epochs", "cf_joint_epochs"):
        assert command[command.index(f"--{key}") + 1] == "1"


def test_toy_config_builds_complete_pipeline_command():
    command = run.build_command("toy", 0, None)
    assert command[command.index("--dataset") + 1] == "toy"
    assert command[command.index("--seeds") + 1] == "1"
    assert command[command.index("--mask_feature_target_max") + 1] == "0.05"
    assert command[command.index("--mask_structure_target_max") + 1] == "0.05"


def test_method_variant_is_fixed_outside_dataset_configs():
    for dataset in ("bail", "pokec_n", "pokec_z", "toy"):
        command = run.build_command(dataset, 0, None)
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
