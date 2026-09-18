# -*- coding: utf-8 -*-
"""Single user-facing entry point for the frozen ML+DOPO V5 FINAL workflow."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FINAL_CORE = "results/scientific_validation/FINAL_core_fixed_baseline_inclusive_5x5"
FINAL_AUX = "results/scientific_validation/FINAL_aux_fixed_baseline_inclusive_5x5"
FINAL_EXPLORATORY = "results/scientific_validation/FINAL_exploratory_fixed_baseline_inclusive_5x5"
FINAL_GROUPING = "results/scientific_validation/FINAL_grouping_sensitivity_5x5"
FINAL_SOURCE = "results/scientific_validation/FINAL_information_source_ablation_5x5"
FINAL_BDE = "results/scientific_validation/FINAL_BDE_paired_ablation_5x5"
FINAL_DELTA = "results/scientific_validation/FINAL_Delta_fixed_5x5"
FINAL_ROW_SENS = "results/scientific_validation/FINAL_row_policy_sensitivity_fixed_5x5"
FINAL_SCREENING_MODE = "results/scientific_validation/FINAL_screening_mode_sensitivity_fixed_5x5"
FINAL_NULL = "results/scientific_validation/FINAL_null_tests"
FINAL_LC = "results/scientific_validation/FINAL_learning_curves"
FINAL_SHAP_CORE = "results/05_Shap/FINAL_core_fixed_baseline_inclusive_5x5"
FINAL_SHAP_AUX = "results/05_Shap/FINAL_aux_fixed_baseline_inclusive_5x5"
FINAL_REVERSE = "results/06_ReverseDesign/FINAL_fixed"
FINAL_EXTERNAL = "results/10_ExternalValidation/FINAL_fixed"


def call(*parts: str) -> None:
    command = [sys.executable, "-u", *parts]
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def scientific(
    tasks: str,
    results: str,
    *,
    bde: str = "without",
    row_policy: str = "baseline_inclusive",
    split: str = "molecule",
    screening: str = "formulation",
    selection_scope: str = "fixed",
    outer: int = 5,
    inner: int = 5,
    models: str = "",
) -> None:
    command = [
        str(ROOT / "08_ScientificValidation" / "run_scientific_evaluation.py"),
        "--tasks", tasks,
        "--results", results,
        "--split-strategies", split,
        "--screening-modes", screening,
        "--bde-modes", bde,
        "--row-policy", row_policy,
        "--selection-scope", selection_scope,
        "--outer-splits", str(outer),
        "--inner-splits", str(inner),
    ]
    if models.strip():
        command.extend(["--models", models])
    call(*command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="Validate frozen data, FINAL protocol, code, and molecule identities")
    sub.add_parser(
        "sync-final-protocol",
        help="Synchronize FINAL protocol metadata in PROJECT_RULES_LOCK without changing CSVs or SHA256",
    )
    sub.add_parser(
        "migrate-release",
        help="LEGACY ONLY: one-time migration for an unmigrated V5 data release",
    )

    p = sub.add_parser("main", help="Run repeated-holdout development models (not formal manuscript metrics)")
    p.add_argument("--tasks", default="ALL")

    p = sub.add_parser("bde-ablation", help="Run development-stage with/without-BDE comparisons")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0,Tg,TS_MPa")

    p = sub.add_parser("nested", help="Run one formal/sensitivity nested validation request")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0,Tg,Char_yield,TS_MPa,FS_MPa")
    p.add_argument("--bde", choices=["without", "with", "both"], default="without")
    p.add_argument("--row-policy", choices=["modified_only", "baseline_inclusive"], default="baseline_inclusive")
    p.add_argument("--selection-scope", choices=["fixed", "curated", "full"], default="fixed")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--results", default="results/scientific_validation/FINAL_all_fixed_baseline_inclusive_5x5")
    p.add_argument("--models", default="")

    sub.add_parser(
        "final-models",
        help="Reproduce frozen FINAL core/auxiliary/exploratory models (do not rerun if already completed unless necessary)",
    )
    sub.add_parser("final-robustness", help="Run FINAL grouping, information-source, BDE, and Delta robustness analyses")
    sub.add_parser("final-interpretation", help="Run FINAL core+auxiliary SHAP and core applicability-domain analysis")
    sub.add_parser("final-screening", help="Run FINAL designed+PubChem combined screening")
    sub.add_parser("external", help="Run independent external validation with frozen FINAL bundles")

    p = sub.add_parser("shap", help="Run FINAL-protocol SHAP stability")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--results", default=FINAL_SHAP_CORE)
    p.add_argument("--parent-results", default="results/05_Shap/FINAL_core_features")

    p = sub.add_parser("split-validation", help="Run FINAL molecule/scaffold/reference grouping sensitivity")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--results", default=FINAL_GROUPING)
    p.add_argument("--models", default="")

    p = sub.add_parser("bde-ablation-scientific", help="Run FINAL paired with/without-BDE nested validation")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0,Tg,TS_MPa")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--results", default=FINAL_BDE)
    p.add_argument("--models", default="")

    sub.add_parser("ad", help="Run FINAL core applicability-domain analysis")
    sub.add_parser("bde-model", help="Run the independent BDE model (not a primary manuscript model)")
    sub.add_parser("diversity", help="Generate dataset diversity summaries")

    p = sub.add_parser("source-ablation", help="Run FINAL information-source ablation")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--results", default=FINAL_SOURCE)
    p.add_argument("--models", default="")

    p = sub.add_parser("screening-mode-ablation", help="Sensitivity: formulation vs molecular information mode under FINAL fixed protocol")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--results", default=FINAL_SCREENING_MODE)

    p = sub.add_parser("row-policy-sensitivity", help="Sensitivity: primary baseline-inclusive vs modified-only subset protocol")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--results", default=FINAL_ROW_SENS)
    p.add_argument("--models", default="")

    p = sub.add_parser("null-tests", help="Run dummy baselines and Y-scrambling diagnostics")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--permutations", type=int, default=999)
    p.add_argument("--model", default="ExtraTrees")
    p.add_argument("--results", default=FINAL_NULL)

    p = sub.add_parser("learning-curves", help="Run molecule-group learning curves")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0,Tg,TS_MPa")
    p.add_argument("--fractions", default="0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--model", default="ExtraTrees")
    p.add_argument("--results", default=FINAL_LC)

    p = sub.add_parser("error-analysis", help="Summarize FINAL outer-test error cases")
    p.add_argument("--results-root", default=FINAL_CORE)
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--output", default="results/scientific_validation/FINAL_outer_error_analysis")

    p = sub.add_parser("paper-audit", help="Audit FINAL figure/table prerequisites and outer-prediction integrity")
    p.add_argument("--results-root", default="results")
    p.add_argument("--output", default="results/09_PaperFigures/audit")

    p = sub.add_parser("paper-main", help="Generate main-text figures from FINAL inputs")
    p.add_argument("--results-root", default="results")
    p.add_argument("--output", default="results/09_PaperFigures/main")

    p = sub.add_parser("paper-supplementary", help="Generate supplementary figures from FINAL inputs")
    p.add_argument("--results-root", default="results")
    p.add_argument("--output", default="results/09_PaperFigures/supplementary")

    p = sub.add_parser("paper-figures", help="Generate main and supplementary figures")
    p.add_argument("--results-root", default="results")
    p.add_argument("--output", default="results/09_PaperFigures")

    p = sub.add_parser("paper-tables", help="Generate paper supplementary tables from FINAL inputs")
    p.add_argument("--results-root", default="results")
    p.add_argument("--output", default="results/09_PaperTables")

    p = sub.add_parser("paper-main-tables", help="Generate main-text tables")
    p.add_argument("--tables-root", default="results/09_PaperTables")
    p.add_argument("--output", default="results/09_MainTextTables")

    p = sub.add_parser("paper-all", help="Audit and generate all FINAL paper figures and tables")
    p.add_argument("--results-root", default="results")
    p.add_argument("--output-root", default="results/09_PaperFigures")
    p.add_argument("--tables-output", default="results/09_PaperTables")
    p.add_argument("--main-tables-output", default="results/09_MainTextTables")

    p = sub.add_parser("candidate-init", help="Extract training DOPO seeds and create the manual candidate template")
    p.add_argument("--top-seeds", type=int, default=20)

    sub.add_parser("candidate-build", help="Validate manual structures and build the DOPO molecule master")

    p = sub.add_parser("candidate-formulations", help="Create FINAL candidate loading/test-condition scenarios")
    p.add_argument("--loadings", default="")
    p.add_argument("--max-loading", type=float, default=30.0)
    p.add_argument("--cone-fluxes", default="35,50")
    p.add_argument("--output", default=f"{FINAL_REVERSE}/candidate_formulation_grid.csv")

    p = sub.add_parser("candidate-screen", help="Predict, evaluate AD, and Pareto-rank with frozen FINAL bundles")
    p.add_argument("--formulations", default=f"{FINAL_REVERSE}/candidate_formulation_grid.csv")
    p.add_argument("--output-dir", default=FINAL_REVERSE)
    p.add_argument("--reliable-threshold", type=float, default=0.70)
    p.add_argument("--caution-threshold", type=float, default=0.50)

    p = sub.add_parser("candidate-all", help="Build, formulate, predict, and rank the designed/manual library")
    p.add_argument("--loadings", default="")
    p.add_argument("--max-loading", type=float, default=30.0)
    p.add_argument("--cone-fluxes", default="35,50")
    p.add_argument("--output-dir", default=FINAL_REVERSE)
    p.add_argument("--reliable-threshold", type=float, default=0.70)
    p.add_argument("--caution-threshold", type=float, default=0.50)

    p = sub.add_parser("combined-screening", help="Rebuild FINAL designed+PubChem 50/35 kW/m² screening and stable priority list")
    p.add_argument("--max-loading", type=float, default=30.0)
    p.add_argument("--reliable-threshold", type=float, default=0.70)
    p.add_argument("--caution-threshold", type=float, default=0.50)
    p.add_argument("--skip-public-rebuild", action="store_true")
    p.add_argument("--skip-screening", action="store_true")

    args = parser.parse_args()

    if args.command == "validate":
        call(str(ROOT / "tools" / "validate_project.py"), "--compile")
        call(str(ROOT / "tools" / "audit_molecule_identity.py"))

    elif args.command == "sync-final-protocol":
        call(str(ROOT / "tools" / "sync_final_protocol_lock.py"))

    elif args.command == "migrate-release":
        call(str(ROOT / "tools" / "migrate_v5_corrected_release.py"))

    elif args.command == "main":
        call(str(ROOT / "run_tasks.py"), "--mode", "main", "--tasks", args.tasks)

    elif args.command == "bde-ablation":
        call(str(ROOT / "run_tasks.py"), "--mode", "bde_ablation", "--tasks", args.tasks)

    elif args.command == "nested":
        scientific(
            args.tasks,
            args.results,
            bde=args.bde,
            row_policy=args.row_policy,
            selection_scope=args.selection_scope,
            outer=args.outer,
            inner=args.inner,
            models=args.models,
        )

    elif args.command == "final-models":
        scientific("LOI,PHRR,THR,UL94_V0", FINAL_CORE)
        scientific("Tg,TS_MPa", FINAL_AUX)
        scientific("Char_yield,FS_MPa", FINAL_EXPLORATORY)

    elif args.command == "final-robustness":
        call(
            str(ROOT / "08_ScientificValidation" / "run_split_validation.py"),
            "--tasks", "LOI,PHRR,THR,UL94_V0",
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", "5",
            "--inner-splits", "5",
            "--results", FINAL_GROUPING,
        )
        call(
            str(ROOT / "08_ScientificValidation" / "run_information_source_ablation.py"),
            "--tasks", "LOI,PHRR,THR,UL94_V0",
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", "5",
            "--inner-splits", "5",
            "--results", FINAL_SOURCE,
        )
        call(
            str(ROOT / "08_ScientificValidation" / "run_bde_comparison_scientific.py"),
            "--tasks", "LOI,PHRR,THR,UL94_V0,Tg,TS_MPa",
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", "5",
            "--inner-splits", "5",
            "--results", FINAL_BDE,
        )
        scientific("Delta_LOI,Delta_PHRR,Delta_THR,Delta_CY", FINAL_DELTA)

    elif args.command == "shap":
        call(
            str(ROOT / "05_Shap" / "run_shap_stability.py"),
            "--tasks", args.tasks,
            "--results", args.results,
            "--parent-results", args.parent_results,
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", "5",
            "--inner-splits", "5",
            "--max-shap-samples", "60",
            "--top-n", "15",
        )

    elif args.command == "final-interpretation":
        call(
            str(ROOT / "05_Shap" / "run_shap_stability.py"),
            "--tasks", "LOI,PHRR,THR,UL94_V0",
            "--results", FINAL_SHAP_CORE,
            "--parent-results", "results/05_Shap/FINAL_core_features",
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", "5",
            "--inner-splits", "5",
            "--max-shap-samples", "60",
            "--top-n", "15",
        )
        call(
            str(ROOT / "05_Shap" / "run_shap_stability.py"),
            "--tasks", "Tg,TS_MPa",
            "--results", FINAL_SHAP_AUX,
            "--parent-results", "results/05_Shap/FINAL_auxiliary_features",
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", "5",
            "--inner-splits", "5",
            "--max-shap-samples", "60",
            "--top-n", "15",
        )
        call(str(ROOT / "08_ScientificValidation" / "run_step4_core_applicability_domain.py"))

    elif args.command == "split-validation":
        command = [
            str(ROOT / "08_ScientificValidation" / "run_split_validation.py"),
            "--tasks", args.tasks,
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", str(args.outer),
            "--inner-splits", str(args.inner),
            "--results", args.results,
        ]
        if args.models.strip():
            command.extend(["--models", args.models])
        call(*command)

    elif args.command == "bde-ablation-scientific":
        command = [
            str(ROOT / "08_ScientificValidation" / "run_bde_comparison_scientific.py"),
            "--tasks", args.tasks,
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", str(args.outer),
            "--inner-splits", str(args.inner),
            "--results", args.results,
        ]
        if args.models.strip():
            command.extend(["--models", args.models])
        call(*command)

    elif args.command == "ad":
        call(str(ROOT / "08_ScientificValidation" / "run_step4_core_applicability_domain.py"))

    elif args.command == "bde-model":
        call(str(ROOT / "07_BDE" / "run_BDE.py"))

    elif args.command == "diversity":
        call(str(ROOT / "tools" / "run_dataset_diversity.py"))

    elif args.command == "source-ablation":
        command = [
            str(ROOT / "08_ScientificValidation" / "run_information_source_ablation.py"),
            "--tasks", args.tasks,
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", str(args.outer),
            "--inner-splits", str(args.inner),
            "--results", args.results,
        ]
        if args.models.strip():
            command.extend(["--models", args.models])
        call(*command)

    elif args.command == "screening-mode-ablation":
        call(
            str(ROOT / "08_ScientificValidation" / "run_screening_mode_ablation.py"),
            "--tasks", args.tasks,
            "--selection-scope", "fixed",
            "--row-policy", "baseline_inclusive",
            "--outer-splits", str(args.outer),
            "--inner-splits", str(args.inner),
            "--results", args.results,
        )

    elif args.command == "row-policy-sensitivity":
        command = [
            str(ROOT / "08_ScientificValidation" / "run_row_policy_sensitivity.py"),
            "--tasks", args.tasks,
            "--selection-scope", "fixed",
            "--outer-splits", str(args.outer),
            "--inner-splits", str(args.inner),
            "--results", args.results,
        ]
        if args.models.strip():
            command.extend(["--models", args.models])
        call(*command)

    elif args.command == "null-tests":
        call(
            str(ROOT / "08_ScientificValidation" / "run_null_tests.py"),
            "--tasks", args.tasks,
            "--permutations", str(args.permutations),
            "--model", args.model,
            "--results", args.results,
        )

    elif args.command == "learning-curves":
        call(
            str(ROOT / "08_ScientificValidation" / "run_learning_curves.py"),
            "--tasks", args.tasks,
            "--fractions", args.fractions,
            "--repeats", str(args.repeats),
            "--model", args.model,
            "--results", args.results,
        )

    elif args.command == "error-analysis":
        call(
            str(ROOT / "08_ScientificValidation" / "analyze_outer_errors.py"),
            "--results-root", args.results_root,
            "--tasks", args.tasks,
            "--top-n", str(args.top_n),
            "--output", args.output,
        )

    elif args.command == "paper-audit":
        call(
            str(ROOT / "09_PaperFigures" / "audit_paper_inputs.py"),
            "--results-root", args.results_root,
            "--output", args.output,
        )

    elif args.command == "paper-main":
        call(
            str(ROOT / "09_PaperFigures" / "make_main_figures.py"),
            "--results-root", args.results_root,
            "--output", args.output,
        )

    elif args.command == "paper-supplementary":
        call(
            str(ROOT / "09_PaperFigures" / "make_supplementary_figures.py"),
            "--results-root", args.results_root,
            "--output", args.output,
        )

    elif args.command == "paper-figures":
        call(
            str(ROOT / "09_PaperFigures" / "make_paper_figures.py"),
            "--results-root", args.results_root,
            "--output", args.output,
        )

    elif args.command == "paper-tables":
        call(
            str(ROOT / "09_PaperFigures" / "build_paper_tables.py"),
            "--results-root", args.results_root,
            "--output", args.output,
        )

    elif args.command == "paper-main-tables":
        call(
            str(ROOT / "09_PaperFigures" / "build_main_text_tables.py"),
            "--tables-root", args.tables_root,
            "--output", args.output,
        )

    elif args.command == "paper-all":
        call(
            str(ROOT / "09_PaperFigures" / "run_all_paper_outputs.py"),
            "--results-root", args.results_root,
            "--output-root", args.output_root,
            "--tables-output", args.tables_output,
            "--main-tables-output", args.main_tables_output,
        )

    elif args.command == "candidate-init":
        call(
            str(ROOT / "06_ReverseDesign" / "initialize_candidate_library.py"),
            "--top-seeds", str(args.top_seeds),
        )

    elif args.command == "candidate-build":
        call(str(ROOT / "06_ReverseDesign" / "build_candidate_master.py"))

    elif args.command == "candidate-formulations":
        call(
            str(ROOT / "06_ReverseDesign" / "build_formulation_grid.py"),
            "--loadings", args.loadings,
            "--max-loading", str(args.max_loading),
            "--cone-fluxes", args.cone_fluxes,
            "--output", args.output,
        )

    elif args.command == "candidate-screen":
        call(
            str(ROOT / "06_ReverseDesign" / "predict_and_rank_candidates.py"),
            "--formulations", args.formulations,
            "--output-dir", args.output_dir,
            "--reliable-threshold", str(args.reliable_threshold),
            "--caution-threshold", str(args.caution_threshold),
        )

    elif args.command == "candidate-all":
        call(str(ROOT / "06_ReverseDesign" / "build_candidate_master.py"))
        output_dir = Path(args.output_dir)
        formulations = output_dir / "candidate_formulation_grid.csv"
        call(
            str(ROOT / "06_ReverseDesign" / "build_formulation_grid.py"),
            "--loadings", args.loadings,
            "--max-loading", str(args.max_loading),
            "--cone-fluxes", args.cone_fluxes,
            "--output", str(formulations),
        )
        call(
            str(ROOT / "06_ReverseDesign" / "predict_and_rank_candidates.py"),
            "--formulations", str(formulations),
            "--output-dir", str(output_dir),
            "--reliable-threshold", str(args.reliable_threshold),
            "--caution-threshold", str(args.caution_threshold),
        )

    elif args.command in {"combined-screening", "final-screening"}:
        command = [
            str(ROOT / "06_ReverseDesign" / "run_combined_screening.py"),
            "--max-loading", str(getattr(args, "max_loading", 30.0)),
            "--reliable-threshold", str(getattr(args, "reliable_threshold", 0.70)),
            "--caution-threshold", str(getattr(args, "caution_threshold", 0.50)),
        ]
        if getattr(args, "skip_public_rebuild", False):
            command.append("--skip-public-rebuild")
        if getattr(args, "skip_screening", False):
            command.append("--skip-screening")
        call(*command)

    elif args.command == "external":
        call(
            str(ROOT / "10_ExternalValidation" / "run_external_validation.py"),
            "--output", FINAL_EXTERNAL,
        )


if __name__ == "__main__":
    main()
