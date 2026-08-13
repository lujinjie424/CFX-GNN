import argparse
import json
import os
import random
import time

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="SCF-GNN reproducible experiment entrypoint")
    parser.add_argument(
        "--encoder",
        type=str,
        default="gcn",
        help="GNN encoder type",
        choices=["gcn", "appnp"],
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cf_risk_self_explainer",
        help="the CFX-GNN method described in the paper",
        choices=["cf_risk_self_explainer"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pokec_z",
        help="dataset name",
        choices=["pokec_n", "pokec_z", "bail", "toy"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="optional dataset-cache directory override",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--smoke_test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--lambda_cons",
        type=float,
        default=0.1,
        help="semantic distillation weight",
    )
    parser.add_argument(
        "--lambda_full",
        type=float,
        default=0.5,
        help="full-graph task loss weight in final multi-view prediction training",
    )
    parser.add_argument(
        "--lambda_cf_pred",
        type=float,
        default=0.1,
        help="counterfactual prediction consistency weight in final multi-view prediction training",
    )
    parser.add_argument(
        "--lambda_fair_pred",
        type=float,
        default=0.5,
        help="soft DP/EO fairness regularization weight in final prediction training",
    )
    parser.add_argument(
        "--selection_min_auc",
        type=float,
        default=None,
        help="minimum validation AUC before checkpoint selection prioritizes fairness",
    )
    parser.add_argument(
        "--lambda_sp_feature",
        type=float,
        default=0.1,
        help="feature-mask sparsity weight",
    )
    parser.add_argument(
        "--lambda_sp_structure",
        type=float,
        default=0.1,
        help="structure-mask sparsity weight",
    )
    parser.add_argument(
        "--min_feature_keep_ratio",
        type=float,
        default=0.05,
        help="minimum non-sensitive feature keep ratio when hard thresholding masks",
    )
    parser.add_argument("--pre_epochs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lambda_indi", type=float, default=1.0)
    parser.add_argument("--lambda_orth", type=float, default=0.05)
    parser.add_argument("--lambda_di_y", type=float, default=0.1)
    parser.add_argument("--lambda_eq", type=float, default=0.01)
    parser.add_argument("--lambda_g", type=float, default=0.001)
    parser.add_argument("--donor_top_k", type=int, default=5)
    parser.add_argument("--donor_tau", type=float, default=0.5)
    parser.add_argument("--donor_fallback_label_penalty", type=float, default=0.25)
    parser.add_argument(
        "--donor_label_score_max_diff",
        type=float,
        default=None,
        help="optional max prediction-score gap for opposite-sensitive donor candidates",
    )
    parser.add_argument(
        "--use_donor_confidence_weight",
        action="store_true",
        default=False,
        help="multiply existing CF-risk weights by donor label-score filter reliability",
    )
    parser.add_argument(
        "--no_donor_confidence_weight",
        dest="use_donor_confidence_weight",
        action="store_false",
    )
    parser.add_argument(
        "--use_donor_reliability_gate",
        action="store_true",
        default=False,
        help="only downweight existing CF-risk weights for donors below a reliability threshold",
    )
    parser.add_argument("--donor_reliability_gate_tau", type=float, default=0.8)
    parser.add_argument("--donor_reliability_gate_floor", type=float, default=0.5)
    parser.add_argument(
        "--donor_require_same_label",
        action="store_true",
        default=False,
        help="for labeled query nodes, prefer opposite-sensitive donors with the same label when enough candidates exist",
    )
    parser.add_argument(
        "--no_donor_require_same_label",
        dest="donor_require_same_label",
        action="store_false",
    )
    parser.add_argument("--donor_select_chunk_size", type=int, default=4096)
    parser.add_argument("--donor_confidence_score_tau", type=float, default=0.05)
    parser.add_argument("--donor_reliability_floor", type=float, default=0.05)
    parser.add_argument(
        "--donor_selection_mode",
        type=str,
        default="distance",
        choices=["distance", "rank_fusion"],
        help="donor ranking rule: raw distance/penalty scoring or rank fusion of z_u and label-score proximity",
    )
    parser.add_argument(
        "--donor_label_rank_weight",
        type=float,
        default=0.25,
        help="label-score rank weight used when donor_selection_mode=rank_fusion",
    )
    parser.add_argument(
        "--mask_ablation_mode",
        type=str,
        default="full",
        choices=["full", "feature_only", "structure_only"],
        help=(
            "masker ablation for bias_clean mode: feature_only disables the bias structure mask, "
            "while structure_only disables the bias feature mask"
        ),
    )
    parser.add_argument(
        "--lambda_bias_margin",
        type=float,
        default=0.0,
        help="optional margin weight encouraging bias-mask risk to exceed clean-graph risk",
    )
    parser.add_argument("--bias_margin", type=float, default=0.0)
    parser.add_argument(
        "--random_mask_trials",
        type=int,
        default=5,
        help="number of matched-sparsity random masks used for the random baseline",
    )
    parser.add_argument("--mask_feature_target_min", type=float, default=None)
    parser.add_argument("--mask_feature_target_max", type=float, default=None)
    parser.add_argument("--mask_structure_target_min", type=float, default=None)
    parser.add_argument("--mask_structure_target_max", type=float, default=None)
    parser.add_argument(
        "--force_retrain_disentangler",
        action="store_true",
        help="ignore cached disentangler checkpoints and retrain encoder+INN",
    )
    parser.add_argument("--num_donors", type=int, default=5)
    parser.add_argument("--use_struct_donor", action="store_true")
    parser.add_argument("--no_struct_donor", action="store_true")
    parser.add_argument("--struct_donor_beta", type=float, default=0.1)
    parser.add_argument("--risk_mode", type=str, default="mean_logit_gap", choices=["mean_logit_gap", "mean_var_logit_gap"])
    parser.add_argument("--risk_tau", type=float, default=0.1)
    parser.add_argument("--risk_temperature", type=float, default=0.05)
    parser.add_argument("--risk_var_gamma", type=float, default=0.0)
    parser.add_argument("--lambda_stage1_map", type=float, default=0.0)
    parser.add_argument("--rho_reduce", type=float, default=0.5)
    parser.add_argument(
        "--repr_reduce_top_ratio",
        type=float,
        default=0.0,
        help="when >0 and reduce_loss_type=repr, compute L_reduce without risk weights on the top-ratio Full-graph representation-shift nodes",
    )
    parser.add_argument("--lambda_reduce", type=float, default=1.0)
    parser.add_argument("--lambda_u_suf", type=float, default=1.0)
    parser.add_argument("--lambda_sp", type=float, default=0.01)
    parser.add_argument("--lambda_mask_budget", type=float, default=0.0)
    parser.add_argument("--mask_feature_budget", type=float, default=None)
    parser.add_argument("--mask_structure_budget", type=float, default=None)
    parser.add_argument("--gsat_stochastic_mask", action="store_true")
    parser.add_argument("--gsat_temperature", type=float, default=0.5)
    parser.add_argument("--lambda_gsat_entropy", type=float, default=0.0)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_ind", type=float, default=1.0)
    parser.add_argument("--lambda_y", type=float, default=1.0)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="encoder output dimension for cf_risk/fairdis-style methods; z_y and z_s each use half",
    )
    parser.add_argument(
        "--warm_y_loss",
        type=str,
        default="mlp_bce",
        choices=["mlp_bce", "linear_bce", "supcon", "balanced_supcon", "cross_s_supcon", "none"],
        help="Phase-1 task-coordinate shaping loss for z_y",
    )
    parser.add_argument(
        "--no_warm_cls",
        action="store_true",
        help="drop the encoder/classifier task loss from CF warm-up; useful for minimal-loss ablations",
    )
    parser.add_argument(
        "--warm_sensitive_loss",
        type=str,
        default="prototype",
        choices=[
            "distance_softplus",
            "distance",
            "conditional_contrastive",
            "contrastive",
            "prototype",
            "label_conditioned_prototype",
            "joo_di",
            "ce",
            "contrastive_ce",
        ],
        help="warm-up objective for z_b: contrastive/prototype geometry by default, optional CE baseline",
    )
    parser.add_argument("--joo_di_y_weight", type=float, default=1.0)
    parser.add_argument("--lambda_s_ce", type=float, default=0.0)
    parser.add_argument("--lambda_zu_con", type=float, default=0.0)
    parser.add_argument("--lambda_zb_y_adv", type=float, default=0.0)
    parser.add_argument("--s_contrast_margin", type=float, default=1.0)
    parser.add_argument("--contrast_temperature", type=float, default=0.07)
    parser.add_argument("--contrast_max_samples", type=int, default=2048)
    parser.add_argument(
        "--no_label_conditioned_zb",
        action="store_true",
        help="use only sensitive labels as positives for z_b contrastive loss",
    )
    parser.add_argument("--lambda_cf", type=float, default=1.0)
    parser.add_argument("--freeze_inn_in_exp", action="store_true", default=True)
    parser.add_argument("--unfreeze_inn_in_exp", dest="freeze_inn_in_exp", action="store_false")
    parser.add_argument("--freeze_encoder_in_exp", action="store_true", default=True)
    parser.add_argument("--unfreeze_encoder_in_exp", dest="freeze_encoder_in_exp", action="store_false")
    parser.add_argument("--encoder_lr_exp", type=float, default=1e-4)
    parser.add_argument("--explainer_lr", type=float, default=1e-3)
    parser.add_argument("--mask_tau", type=float, default=1.0)
    parser.add_argument(
        "--lambda_zs_fair",
        type=float,
        default=0.0,
        help="weight for z_s-sensitive exposure regularization when prediction uses z_y",
    )
    parser.add_argument("--static_exp_condition", action="store_true")
    parser.add_argument("--local_top_k", type=int, default=512)
    parser.add_argument("--local_hops", type=int, default=2)
    parser.add_argument(
        "--explainer_scope",
        type=str,
        default="local",
        choices=["local", "all_nodes"],
        help="train explanation losses and masks on a selected local region or the full transductive graph",
    )
    parser.add_argument(
        "--joint_mask_mode",
        type=str,
        default="soft",
        choices=["soft", "straight_through_hard"],
        help="mask distribution used in Phase-3 joint refinement",
    )
    parser.add_argument("--fair_early_stop", action="store_true")
    parser.add_argument("--fair_stop_min_auc", type=float, default=0.0)
    parser.add_argument("--fair_stop_cf_weight", type=float, default=1.0)
    parser.add_argument("--fair_stop_logit_weight", type=float, default=0.5)
    parser.add_argument("--fair_stop_auc_weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint_selection_mode",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Phase-3 checkpoint selection metric source; hard remains available as a diagnostic gate",
    )
    parser.add_argument(
        "--soft_select_min_auc",
        type=float,
        default=None,
        help="optional validation AUC floor for soft Phase-3 checkpoint updates; keeps the pre-joint state if no candidate passes",
    )
    parser.add_argument(
        "--hard_select_min_auc",
        type=float,
        default=None,
        help="minimum hard-mask validation AUC for explanation checkpoint selection; defaults to selection_min_auc or 0.55",
    )
    parser.add_argument(
        "--hard_select_min_margin",
        type=float,
        default=0.01,
        help="minimum hard-mask validation AUC gap between explanation and complement for checkpoint selection",
    )
    parser.add_argument(
        "--joint_early_stop_patience",
        type=int,
        default=0,
        help="stop Phase 3 after this many validation checks without checkpoint improvement; 0 disables",
    )
    parser.add_argument(
        "--joint_early_stop_min_delta",
        type=float,
        default=1e-4,
        help="minimum checkpoint-score improvement counted by Phase-3 early stopping",
    )
    parser.set_defaults(
        cf_stage1_regularizer="orth",
        phase2_mask_role="bias_clean",
        cf_ref_mode="local_quantile",
        stage1_map_loss="none",
        reduce_loss_type="repr",
        lambda_pred_exp=0.0,
        final_pred_source="zy",
        lambda_h_cf_inv=0.0,
        lambda_local_keep=0.0,
        cf_group_calibration=False,
        phase3_update_mode="clean_head_only",
        calib_epochs=0,
    )
    parser.add_argument("--calib_lr", type=float, default=0.001)
    parser.add_argument("--lambda_calib_ind", type=float, default=0.5)
    parser.add_argument("--lambda_calib_group", type=float, default=0.5)
    parser.add_argument("--lambda_calib_distill", type=float, default=0.1)
    parser.add_argument("--lambda_calib_pair", type=float, default=0.3)
    parser.add_argument("--calib_auc_drop_tol", type=float, default=0.01)
    parser.add_argument("--calib_min_auc", type=float, default=0.0)
    parser.add_argument("--calib_select_cf_weight", type=float, default=1.0)
    parser.add_argument("--calib_select_logit_weight", type=float, default=0.5)
    parser.add_argument("--calib_select_auc_weight", type=float, default=0.0)
    parser.add_argument("--calib_hard_guard", action="store_true")
    parser.add_argument("--calib_guard_min_auc", type=float, default=0.55)
    parser.add_argument("--calib_guard_min_margin", type=float, default=0.0)
    parser.add_argument(
        "--strict_run_validity",
        action="store_true",
        help="mark the run as an objective failure when final hard-mask validity criteria are not met",
    )
    parser.add_argument(
        "--validity_min_explain_auc",
        type=float,
        default=0.55,
        help="minimum final hard explanation AUC used in hard-mask validity diagnostics",
    )
    parser.add_argument(
        "--validity_min_exp_comp_gap",
        type=float,
        default=0.0,
        help="minimum final AUC gap between hard explanation and complement validity diagnostics",
    )
    parser.add_argument(
        "--validity_max_cf_flip",
        type=float,
        default=None,
        help="optional maximum final hard explanation CF flip used in hard-mask validity diagnostics",
    )
    parser.add_argument("--cf_warm_epochs", type=int, default=None)
    parser.add_argument("--cf_exp_epochs", type=int, default=None)
    parser.add_argument("--cf_joint_epochs", type=int, default=None)
    parser.add_argument("--no_risk_weight", action="store_true")
    parser.add_argument("--single_donor", action="store_true")
    parser.add_argument("--fixed_cf_target", action="store_true")
    parser.add_argument("--no_u_suf", action="store_true")
    parser.add_argument("--no_reduce_loss", action="store_true")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="optional checkpoint root; defaults to <repo>/ckpt",
    )
    parser.add_argument(
        "--case_study_output",
        type=str,
        default=None,
        help="optional Markdown path for exporting learned-mask case-study nodes",
    )
    parser.add_argument(
        "--export_mask_scores",
        type=str,
        default=None,
        help="optional npz output path for soft feature/edge bias scores; supports {seed}",
    )
    parser.add_argument(
        "--export_explanation_artifacts",
        type=str,
        default=None,
        help="optional npz path for the embeddings used by the manuscript RQ2 visualization; supports {seed} and {dataset}",
    )
    parser.add_argument("--case_top_nodes", type=int, default=2)
    parser.add_argument("--case_top_features", type=int, default=5)
    parser.add_argument("--case_top_neighbors", type=int, default=5)
    parser.add_argument(
        "--case_selection_mode",
        type=str,
        default="top",
        choices=["top", "sensitive_diverse", "feature_diverse"],
        help="case-study node selection rule",
    )
    parser.add_argument(
        "--case_candidate_pool",
        type=int,
        default=2000,
        help="maximum number of test nodes considered before diverse case-study selection",
    )
    parser.add_argument(
        "--case_min_feature_soft",
        type=float,
        default=0.05,
        help="preferred minimum top non-sensitive feature-mask value for feature-diverse cases",
    )
    return parser.parse_args()


class Config(object):
    def __init__(self, args):
        abs_dir = os.path.dirname(os.path.realpath(__file__))
        if args.data_dir:
            data_dir = os.path.abspath(args.data_dir)
        elif args.dataset in ("pokec_z", "pokec_n"):
            data_dir = os.path.join(abs_dir, "dataset", "pokec")
        else:
            data_dir = os.path.join(abs_dir, "dataset", args.dataset)

        self.method = args.method
        self.encoder_type = args.encoder
        self.dataset = args.dataset
        self.abs_dir = abs_dir
        self.data_dir = data_dir
        self.gpu = args.gpu
        self.index = None
        self.graph_path = f"{data_dir}/{args.dataset}_graph.bin"
        self.index_path = f"{data_dir}/{args.dataset}_index.bin"
        self.adj_path = f"{data_dir}/{args.dataset}_adj_csr.npz"
        self.check_dataset()
        self.ckpt_dir = args.ckpt_dir or os.path.join(abs_dir, "ckpt")
        self.case_study_output = args.case_study_output
        self.export_mask_scores = args.export_mask_scores
        self.export_explanation_artifacts = args.export_explanation_artifacts
        self.case_top_nodes = args.case_top_nodes
        self.case_top_features = args.case_top_features
        self.case_top_neighbors = args.case_top_neighbors
        self.case_selection_mode = args.case_selection_mode
        self.case_candidate_pool = args.case_candidate_pool
        self.case_min_feature_soft = args.case_min_feature_soft
        self.lr = args.lr
        self.lambda_cons = args.lambda_cons
        self.lambda_full = args.lambda_full
        self.lambda_cf_pred = args.lambda_cf_pred
        self.lambda_fair_pred = args.lambda_fair_pred
        self.selection_min_auc = args.selection_min_auc
        self.lambda_sp_feature = args.lambda_sp_feature
        self.lambda_sp_structure = args.lambda_sp_structure
        self.min_feature_keep_ratio = args.min_feature_keep_ratio
        self.lambda_indi = args.lambda_indi
        self.lambda_orth = args.lambda_orth
        self.cf_stage1_regularizer = args.cf_stage1_regularizer
        self.lambda_di_y = args.lambda_di_y
        self.lambda_eq = args.lambda_eq
        self.lambda_g = args.lambda_g
        self.donor_top_k = args.donor_top_k
        self.donor_tau = args.donor_tau
        self.donor_fallback_label_penalty = args.donor_fallback_label_penalty
        self.donor_label_score_max_diff = args.donor_label_score_max_diff
        self.use_donor_confidence_weight = args.use_donor_confidence_weight
        self.use_donor_reliability_gate = args.use_donor_reliability_gate
        self.donor_reliability_gate_tau = args.donor_reliability_gate_tau
        self.donor_reliability_gate_floor = args.donor_reliability_gate_floor
        self.donor_require_same_label = args.donor_require_same_label
        self.donor_select_chunk_size = args.donor_select_chunk_size
        self.donor_confidence_score_tau = args.donor_confidence_score_tau
        self.donor_reliability_floor = args.donor_reliability_floor
        self.donor_selection_mode = args.donor_selection_mode
        self.donor_label_rank_weight = args.donor_label_rank_weight
        self.phase2_mask_role = args.phase2_mask_role
        self.mask_ablation_mode = args.mask_ablation_mode
        self.lambda_bias_margin = args.lambda_bias_margin
        self.bias_margin = args.bias_margin
        self.random_mask_trials = args.random_mask_trials
        self.mask_feature_target_min = args.mask_feature_target_min
        self.mask_feature_target_max = args.mask_feature_target_max
        self.mask_structure_target_min = args.mask_structure_target_min
        self.mask_structure_target_max = args.mask_structure_target_max
        self.force_retrain_disentangler = args.force_retrain_disentangler
        self.num_donors = 1 if args.single_donor else args.num_donors
        self.use_struct_donor = args.use_struct_donor
        self.no_struct_donor = args.no_struct_donor
        self.struct_donor_beta = args.struct_donor_beta
        self.risk_mode = args.risk_mode
        self.risk_tau = args.risk_tau
        self.risk_temperature = args.risk_temperature
        self.risk_var_gamma = args.risk_var_gamma
        self.cf_ref_mode = args.cf_ref_mode
        self.stage1_map_loss = args.stage1_map_loss
        self.lambda_stage1_map = args.lambda_stage1_map
        self.rho_reduce = args.rho_reduce
        self.repr_reduce_top_ratio = args.repr_reduce_top_ratio
        self.reduce_loss_type = args.reduce_loss_type
        self.lambda_reduce = args.lambda_reduce
        self.lambda_u_suf = args.lambda_u_suf
        self.lambda_pred_exp = args.lambda_pred_exp
        self.lambda_sp = args.lambda_sp
        self.lambda_mask_budget = args.lambda_mask_budget
        self.mask_feature_budget = args.mask_feature_budget
        self.mask_structure_budget = args.mask_structure_budget
        self.gsat_stochastic_mask = args.gsat_stochastic_mask
        self.gsat_temperature = args.gsat_temperature
        self.lambda_gsat_entropy = args.lambda_gsat_entropy
        self.lambda_rec = args.lambda_rec
        self.lambda_ind = args.lambda_ind
        self.lambda_y = args.lambda_y
        self.lambda_s = args.lambda_s
        self.hidden_dim = args.hidden_dim
        self.warm_y_loss = args.warm_y_loss
        self.final_pred_source = args.final_pred_source
        self.no_warm_cls = args.no_warm_cls
        self.warm_sensitive_loss = args.warm_sensitive_loss
        self.joo_di_y_weight = args.joo_di_y_weight
        self.lambda_s_ce = args.lambda_s_ce
        self.lambda_zu_con = args.lambda_zu_con
        self.lambda_zb_y_adv = args.lambda_zb_y_adv
        self.s_contrast_margin = args.s_contrast_margin
        self.contrast_temperature = args.contrast_temperature
        self.contrast_max_samples = args.contrast_max_samples
        self.no_label_conditioned_zb = args.no_label_conditioned_zb
        self.lambda_cf = args.lambda_cf
        self.freeze_inn_in_exp = args.freeze_inn_in_exp
        self.freeze_encoder_in_exp = args.freeze_encoder_in_exp
        self.encoder_lr_exp = args.encoder_lr_exp
        self.explainer_lr = args.explainer_lr
        self.mask_tau = args.mask_tau
        self.lambda_h_cf_inv = args.lambda_h_cf_inv
        self.lambda_zs_fair = args.lambda_zs_fair
        self.static_exp_condition = args.static_exp_condition
        self.local_top_k = args.local_top_k
        self.local_hops = args.local_hops
        self.explainer_scope = args.explainer_scope
        self.lambda_local_keep = args.lambda_local_keep
        self.joint_mask_mode = args.joint_mask_mode
        self.fair_early_stop = args.fair_early_stop
        self.fair_stop_min_auc = args.fair_stop_min_auc
        self.fair_stop_cf_weight = args.fair_stop_cf_weight
        self.fair_stop_logit_weight = args.fair_stop_logit_weight
        self.fair_stop_auc_weight = args.fair_stop_auc_weight
        self.checkpoint_selection_mode = args.checkpoint_selection_mode
        self.soft_select_min_auc = args.soft_select_min_auc
        self.hard_select_min_auc = args.hard_select_min_auc
        self.hard_select_min_margin = args.hard_select_min_margin
        self.joint_early_stop_patience = args.joint_early_stop_patience
        self.joint_early_stop_min_delta = args.joint_early_stop_min_delta
        self.cf_group_calibration = args.cf_group_calibration
        self.phase3_update_mode = args.phase3_update_mode
        self.calib_epochs = args.calib_epochs
        self.calib_lr = args.calib_lr
        self.lambda_calib_ind = args.lambda_calib_ind
        self.lambda_calib_group = args.lambda_calib_group
        self.lambda_calib_distill = args.lambda_calib_distill
        self.lambda_calib_pair = args.lambda_calib_pair
        self.calib_auc_drop_tol = args.calib_auc_drop_tol
        self.calib_min_auc = args.calib_min_auc
        self.calib_select_cf_weight = args.calib_select_cf_weight
        self.calib_select_logit_weight = args.calib_select_logit_weight
        self.calib_select_auc_weight = args.calib_select_auc_weight
        self.calib_hard_guard = args.calib_hard_guard
        self.calib_guard_min_auc = args.calib_guard_min_auc
        self.calib_guard_min_margin = args.calib_guard_min_margin
        self.strict_run_validity = args.strict_run_validity
        self.validity_min_explain_auc = args.validity_min_explain_auc
        self.validity_min_exp_comp_gap = args.validity_min_exp_comp_gap
        self.validity_max_cf_flip = args.validity_max_cf_flip
        self.cf_warm_epochs = args.cf_warm_epochs
        self.cf_exp_epochs = args.cf_exp_epochs
        self.cf_joint_epochs = args.cf_joint_epochs
        self.no_risk_weight = args.no_risk_weight
        self.single_donor = args.single_donor
        self.fixed_cf_target = args.fixed_cf_target
        self.no_u_suf = args.no_u_suf
        self.no_reduce_loss = args.no_reduce_loss
        self.pre_epochs = args.pre_epochs
        self.epochs = args.epochs

    def check_dataset(self):
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Missing prepared graph cache: {self.graph_path}")

    def set_seed(self, seed):
        self.seed = seed
        self.encoder_path = f"{self.ckpt_dir}/{self.dataset}/{self.encoder_type}-seed-{seed}-pretrain.pt"


def setup_seed(seed):
    import dgl

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def experiment_seeds(smoke_test=False):
    return [1] if smoke_test else list(range(1, 6))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


def summarize_array(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def metric_block(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def flatten_result(result):
    flat = {
        "dataset": result["dataset"],
        "encoder": result["encoder"],
        "method": result["method"],
        "seed": result["seed"],
        "lr": result["lr"],
        "pre_epochs": result["pre_epochs"],
        "epochs": result["epochs"],
        "runtime_sec": result["runtime_sec"],
        "checkpoint_dir": result["checkpoint_dir"],
    }
    flat.update(metric_block("run_env", result.get("run_env", {})))
    flat.update(metric_block("explain", result["explain_graph"]))
    flat.update(metric_block("original", result["original_graph"]))
    flat.update(metric_block("complement", result["complement_graph"]))
    if "bias_graph" in result:
        flat.update(metric_block("bias", result["bias_graph"]))
    if "clean_graph" in result:
        flat.update(metric_block("clean", result["clean_graph"]))
    flat.update(metric_block("system_fairness", result["system_fairness"]))
    flat.update(metric_block("fid_accuracy", result["fid_accuracy"]))
    flat.update(metric_block("fid_cf", result["fid_cf"]))
    diagnostics = result.get("diagnostics", {})
    flat.update(metric_block("mask", diagnostics.get("mask_sparsity", {})))
    flat.update(metric_block("complement_mask", diagnostics.get("complement_mask_sparsity", {})))
    flat.update(metric_block("cf_validity", diagnostics.get("cf_validity", {})))
    flat.update(metric_block("hard_mask_validity", diagnostics.get("hard_mask_validity", {})))
    flat.update(metric_block("bias_clean", diagnostics.get("bias_clean", {})))
    flat.update(metric_block("random_mask", diagnostics.get("random_mask", {})))
    flat.update(metric_block("run_status", diagnostics.get("run_status", {})))
    flat.update(metric_block("cf_risk", diagnostics.get("cf_risk", {})))
    flat.update(metric_block("high_risk_subset", diagnostics.get("high_risk_subset", {})))
    flat.update(metric_block("disentangle", diagnostics.get("disentangle_diagnostics", {})))
    flat.update(metric_block("run_config", diagnostics.get("run_config", {})))
    flat.update(metric_block("donor_selector", diagnostics.get("donor_selector", {})))
    flat.update(metric_block("donor_reliability", diagnostics.get("donor_reliability", {})))
    flat.update(metric_block("donor_reliability_mask", diagnostics.get("donor_reliability_mask", {})))
    return flat


def make_jsonable(value):
    if isinstance(value, dict):
        return {key: make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def print_metric_summary(title, arrays, scale=100.0):
    print(f"\n================ {title} ================")
    for name, values in arrays.items():
        print(name + ":", round(np.mean(values) * scale, 2), "±", round(np.std(values) * scale, 2), sep="")


def main():
    from train import train_cf_risk_self_explainer

    if torch.cuda.is_available():
        if cfg.gpu < 0 or cfg.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu={cfg.gpu} is invalid; visible CUDA device count is {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(cfg.gpu)

    print(f"Dataset: {cfg.dataset}, Encoder:{cfg.encoder_type}, Method:{cfg.method}")
    run_env = {
        "device": f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_id": int(cfg.gpu),
        "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        run_env["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Run environment: {run_env}")
    seeds = experiment_seeds(args.smoke_test)
    num_runs = len(seeds)
    run_results = []

    total_acc, total_f1, total_auc_roc, total_parity, total_equality = [np.zeros(num_runs) for _ in range(5)]
    total_acc_origin, total_f1_origin, total_auc_roc_origin, total_parity_origin, total_equality_origin = [
        np.zeros(num_runs) for _ in range(5)
    ]
    total_acc_comp, total_f1_comp, total_auc_roc_comp, total_parity_comp, total_equality_comp = [
        np.zeros(num_runs) for _ in range(5)
    ]
    total_original_us, total_explain_us, total_comp_us, total_original_flip, total_explain_flip, total_comp_flip = [
        np.zeros(num_runs) for _ in range(6)
    ]
    Fid_plus_flip, Fid_minus_flip, Fid_plus_prob, Fid_minus_prob = [np.zeros(num_runs) for _ in range(4)]
    CF_fid_plus_prob, CF_fid_minus_prob, CF_fid_plus_flip, CF_fid_minus_flip = [
        np.zeros(num_runs) for _ in range(4)
    ]

    for count, seed in enumerate(seeds):
        setup_seed(seed)
        cfg.set_seed(seed)
        print(f"===========seed: {seed}===========")

        trainer = train_cf_risk_self_explainer
        print(f"Train {cfg.method} with {trainer.__name__}.py...")
        run_start = time.time()
        train_output = trainer.train(cfg)
        if len(train_output) == 6:
            best_result, best_result_origin, best_result_comp, system_fairness, fid_accuracy, fid_cf = train_output
            diagnostics = {}
        else:
            (
                best_result,
                best_result_origin,
                best_result_comp,
                system_fairness,
                fid_accuracy,
                fid_cf,
                diagnostics,
            ) = train_output
        runtime_sec = time.time() - run_start

        result_record = {
            "dataset": cfg.dataset,
            "encoder": cfg.encoder_type,
            "method": cfg.method,
            "seed": seed,
            "lr": cfg.lr,
            "pre_epochs": cfg.pre_epochs,
            "epochs": cfg.epochs,
            "runtime_sec": runtime_sec,
            "checkpoint_dir": f"{cfg.ckpt_dir}/{cfg.dataset}/Disentangler/{cfg.encoder_type}/{cfg.seed}",
            "run_env": run_env,
            "explain_graph": best_result,
            "original_graph": best_result_origin,
            "complement_graph": best_result_comp,
            "system_fairness": system_fairness,
            "fid_accuracy": fid_accuracy,
            "fid_cf": fid_cf,
            "diagnostics": diagnostics,
        }
        result_record["bias_graph"] = best_result
        result_record["clean_graph"] = best_result_comp
        run_results.append(result_record)

        total_acc[count] = best_result["acc"]
        total_f1[count] = best_result["F1"]
        total_auc_roc[count] = best_result["auc_roc"]
        total_parity[count] = best_result["parity"]
        total_equality[count] = best_result["equality"]

        total_acc_origin[count] = best_result_origin["acc"]
        total_f1_origin[count] = best_result_origin["F1"]
        total_auc_roc_origin[count] = best_result_origin["auc_roc"]
        total_parity_origin[count] = best_result_origin["parity"]
        total_equality_origin[count] = best_result_origin["equality"]

        total_acc_comp[count] = best_result_comp["acc"]
        total_f1_comp[count] = best_result_comp["F1"]
        total_auc_roc_comp[count] = best_result_comp["auc_roc"]
        total_parity_comp[count] = best_result_comp["parity"]
        total_equality_comp[count] = best_result_comp["equality"]

        total_original_us[count] = system_fairness["original_us"]
        total_explain_us[count] = system_fairness["explain_us"]
        total_comp_us[count] = system_fairness["comp_us"]
        total_original_flip[count] = system_fairness["original_flip"]
        total_explain_flip[count] = system_fairness["explain_flip"]
        total_comp_flip[count] = system_fairness["comp_flip"]

        Fid_plus_flip[count] = fid_accuracy["Fid_plus_flip"]
        Fid_minus_flip[count] = fid_accuracy["Fid_minus_flip"]
        Fid_plus_prob[count] = fid_accuracy["Fid_plus_prob"]
        Fid_minus_prob[count] = fid_accuracy["Fid_minus_prob"]

        CF_fid_plus_prob[count] = fid_cf["CF_fid_plus_prob"]
        CF_fid_minus_prob[count] = fid_cf["CF_fid_minus_prob"]
        CF_fid_plus_flip[count] = fid_cf["CF_fid_plus_flip"]
        CF_fid_minus_flip[count] = fid_cf["CF_fid_minus_flip"]

    print_metric_summary(
        "Performance Evaluation on Explain Graph",
        {"Acc": total_acc, "f1": total_f1, "Auc": total_auc_roc, "parity": total_parity, "equality": total_equality},
    )
    print_metric_summary(
        "Performance Evaluation on Original Graph",
        {
            "Acc": total_acc_origin,
            "f1": total_f1_origin,
            "Auc": total_auc_roc_origin,
            "parity": total_parity_origin,
            "equality": total_equality_origin,
        },
    )
    print_metric_summary(
        "Performance Evaluation on Complementary Graph",
        {
            "Acc": total_acc_comp,
            "f1": total_f1_comp,
            "Auc": total_auc_roc_comp,
            "parity": total_parity_comp,
            "equality": total_equality_comp,
        },
    )
    print_metric_summary(
        "System Fairness",
        {
            "Unfairness Score on original Graph": total_original_us,
            "Flip Rate on original Graph": total_original_flip,
            "Unfairness Score on Explain Graph": total_explain_us,
            "Flip Rate on Explain Graph": total_explain_flip,
            "Unfairness Score on Complementary Graph": total_comp_us,
            "Flip Rate on Complementary Graph": total_comp_flip,
        },
    )
    print_metric_summary(
        "Fid on Accuracy",
        {
            "Fid+ on prob": Fid_plus_prob,
            "Fid- on prob": Fid_minus_prob,
            "Fid+ on flip": Fid_plus_flip,
            "Fid- on flip": Fid_minus_flip,
        },
    )
    print_metric_summary(
        "Fid on CF",
        {
            "Fid+ on prob": CF_fid_plus_prob,
            "Fid- on prob": CF_fid_minus_prob,
            "Fid+ on flip": CF_fid_plus_flip,
            "Fid- on flip": CF_fid_minus_flip,
        },
    )

    summary = {
        "dataset": cfg.dataset,
        "encoder": cfg.encoder_type,
        "method": cfg.method,
        "seeds": seeds,
        "run_env": run_env,
        "explain_graph": {
            "acc": summarize_array(total_acc),
            "F1": summarize_array(total_f1),
            "auc_roc": summarize_array(total_auc_roc),
            "parity": summarize_array(total_parity),
            "equality": summarize_array(total_equality),
        },
        "original_graph": {
            "acc": summarize_array(total_acc_origin),
            "F1": summarize_array(total_f1_origin),
            "auc_roc": summarize_array(total_auc_roc_origin),
            "parity": summarize_array(total_parity_origin),
            "equality": summarize_array(total_equality_origin),
        },
        "complement_graph": {
            "acc": summarize_array(total_acc_comp),
            "F1": summarize_array(total_f1_comp),
            "auc_roc": summarize_array(total_auc_roc_comp),
            "parity": summarize_array(total_parity_comp),
            "equality": summarize_array(total_equality_comp),
        },
        "system_fairness": {
            "original_us": summarize_array(total_original_us),
            "explain_us": summarize_array(total_explain_us),
            "comp_us": summarize_array(total_comp_us),
            "original_flip": summarize_array(total_original_flip),
            "explain_flip": summarize_array(total_explain_flip),
            "comp_flip": summarize_array(total_comp_flip),
        },
        "fid_accuracy": {
            "Fid_plus_flip": summarize_array(Fid_plus_flip),
            "Fid_minus_flip": summarize_array(Fid_minus_flip),
            "Fid_plus_prob": summarize_array(Fid_plus_prob),
            "Fid_minus_prob": summarize_array(Fid_minus_prob),
        },
        "fid_cf": {
            "CF_fid_plus_prob": summarize_array(CF_fid_plus_prob),
            "CF_fid_minus_prob": summarize_array(CF_fid_minus_prob),
            "CF_fid_plus_flip": summarize_array(CF_fid_plus_flip),
            "CF_fid_minus_flip": summarize_array(CF_fid_minus_flip),
        },
    }
    summary["bias_graph"] = summary["explain_graph"]
    summary["clean_graph"] = summary["complement_graph"]
    paper_blocks = [result.get("diagnostics", {}).get("paper_explanation_eval") for result in run_results]
    paper_blocks = [block for block in paper_blocks if block]
    if paper_blocks:
        summary["paper_explanation_eval"] = {}
        for view in ("full", "clean", "bias"):
            summary["paper_explanation_eval"][view] = {}
            keys = paper_blocks[0][view].keys()
            for key in keys:
                values = [block[view][key] for block in paper_blocks if block[view][key] is not None]
                summary["paper_explanation_eval"][view][key] = summarize_array(values) if values else None
    public_summary = {
        "dataset": summary["dataset"],
        "encoder": summary["encoder"],
        "method": "CFX-GNN",
        "seeds": summary["seeds"],
        "full_clean_bias": summary.get("paper_explanation_eval", {}),
    }
    print("\n================ Structured Summary ================")
    print(json.dumps(make_jsonable(public_summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args)
    main()
