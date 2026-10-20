"""Local commands. No background training or automatic server reconfiguration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import runtime  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = "configs/study_001"


def read_config(path):
    path=Path(path)
    if not path.is_absolute():
        path=ROOT/path
    return json.loads(path.read_text())


def split_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def study_run(args, parser):
    from .study import run_study
    from .study_panels import load_study
    study=load_study(ROOT/STUDY_DIR)
    protocol=study["protocol"]
    defaults=protocol["single_map"] if args.single_map else {
        "steps":protocol["pilot"]["steps_per_task"],"eval_every":protocol["pilot"]["eval_every"]}
    steps=defaults["steps"] if args.steps is None else args.steps
    eval_every=defaults["eval_every"] if args.eval_every is None else args.eval_every
    if steps < 0 or eval_every < 1:
        parser.error("--steps cannot be negative and --eval-every must be positive")
    result=run_study(study,read_config(args.arch),args.seed,split_list(args.tasks),
                     steps=steps,eval_every=eval_every,output=args.output,root=ROOT,
                     single_map=args.single_map,final_evaluation=args.final_evaluation,
                     tracking=protocol.get("tracking"))
    keys=("status","output","phase","architecture_id","init_seed","task_keys","acquisition",
          "auc","retention","single_map_check","interactions","elapsed_seconds","mlflow")
    return {key:result[key] for key in keys if key in result}


def study_pilot(args, parser):
    archs,tasks,seeds=split_list(args.archs),split_list(args.tasks),split_list(args.seeds)
    for arch in archs:
        if not (ROOT/STUDY_DIR/f"arch-{arch}.json").is_file():
            parser.error(f"Unknown study architecture {arch}")
    extra=[]
    if args.steps is not None:
        extra+=["--steps",str(args.steps)]
    if args.eval_every is not None:
        extra+=["--eval-every",str(args.eval_every)]
    for arch in archs:
        for task in tasks:
            for seed in seeds:
                subprocess.run([sys.executable,"-B","-m","ucl.cli","study-run",
                                "--arch",f"{STUDY_DIR}/arch-{arch}.json","--seed",str(int(seed)),
                                "--tasks",task,*extra],cwd=ROOT,check=True)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(prog="ucl")
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ("train","evaluate","resume"):
        command=commands.add_parser(name)
        command.add_argument("--config",default="configs/EXP-002.json")
        command.add_argument("--output")
        command.add_argument("--task")
        command.add_argument("--curriculum")
        command.add_argument("--seed",type=int)
        command.add_argument("--steps",type=int)
        command.add_argument("--checkpoint",required=name=="resume")
    dashboard=commands.add_parser("dashboard")
    dashboard.add_argument("--port",type=int,default=8501)
    compare=commands.add_parser("compare")
    compare.add_argument("runs",nargs="+")
    compare.add_argument("--output",default="runs/comparison.json")
    transfer=commands.add_parser("learning-to-learn")
    transfer.add_argument("--config",default="configs/EXP-002.json")
    transfer.add_argument("--seed",type=int,default=7)
    transfer.add_argument("--output")
    matrix=commands.add_parser("matrix")
    matrix.add_argument("--seeds",default="7,19")
    matrix.add_argument("--include-llm",action="store_true")
    build=commands.add_parser("study-build")
    build.add_argument("--force",action="store_true")
    run=commands.add_parser("study-run")
    run.add_argument("--arch",default=f"{STUDY_DIR}/arch-LSTM.json")
    run.add_argument("--seed",type=int,default=7)
    run.add_argument("--tasks",default="A")
    run.add_argument("--steps",type=int)
    run.add_argument("--eval-every",type=int)
    run.add_argument("--single-map",action="store_true")
    run.add_argument("--final-evaluation",action="store_true")
    run.add_argument("--output")
    pilot=commands.add_parser("study-pilot")
    pilot.add_argument("--archs",default="LSTM,MLP")
    pilot.add_argument("--tasks",default="A,B,C")
    pilot.add_argument("--seeds",default="7,19,31")
    pilot.add_argument("--steps",type=int)
    pilot.add_argument("--eval-every",type=int)
    record=commands.add_parser("study-record")
    record.add_argument("--run",required=True,
                        help="Study run folder relative to runs/, e.g. study_001/<folder>")
    record.add_argument("--checkpoint",required=True,help="A checkpoint named in its summary")
    record.add_argument("--task",required=True,help="Task key of the study, e.g. B")
    record.add_argument("--panel",required=True,help="Panel name; sealed panels are refused")
    record.add_argument("--maps",required=True,help="1-based map numbers, e.g. 1,2,3")
    record.add_argument("--output",required=True)
    reevaluate=commands.add_parser("study-reevaluate")
    reevaluate.add_argument("--run",required=True,
                            help="Finished study run folder relative to runs/, "
                                 "e.g. study_001/<folder>")
    reevaluate.add_argument("--replicas",type=int,default=3,
                            help="New sampling seeds per map (default 3)")
    chain=commands.add_parser("study-chain")
    chain.add_argument("--plan",required=True,help="Preregistered plan, e.g. plans/<name>.json")
    chain.add_argument("--retry",action="store_true",
                       help="Repeat the FAILED or INTERRUPTED item with the same seed")
    chain.add_argument("--limit",type=int,help="Stop after this many completed items")
    chain.add_argument("--status",action="store_true",help="Only show where the chain stands")
    sealed_build=commands.add_parser("study-sealed-build")
    sealed_build.add_argument("--name",required=True)
    sealed_build.add_argument("--study",default="configs/study_001")
    sealed_build.add_argument("--task",required=True)
    sealed_build.add_argument("--base",type=int,required=True,help="First map seed to scan")
    sealed_build.add_argument("--size",type=int,default=64)
    sealed_open=commands.add_parser("study-sealed-open")
    sealed_open.add_argument("--spec",required=True,
                             help="Frozen analysis spec with a 'sealed' block")
    probe=commands.add_parser("study-probe-time")
    probe.add_argument("--plan",required=True)
    probe.add_argument("--labels",required=True,help="Comma-separated arm labels to probe")
    probe.add_argument("--node",default="memory",help="Architecture node whose output is read")
    probe.add_argument("--checkpoint",default="best_B")
    probe.add_argument("--replicas",type=int,default=2)
    probe.add_argument("--id",default="C01-SONDA-TIEMPO")
    contrasts=commands.add_parser("study-contrasts")
    contrasts.add_argument("--spec",required=True,
                           help="Analysis spec with plans, scores and contrast families")
    variance=commands.add_parser("study-variance")
    variance.add_argument("--plan",required=True,help="Finished plan, e.g. plans/<name>.json")
    variance.add_argument("--bootstrap",type=int,default=1000,
                          help="Seed resamples for 95%% intervals (default 1000)")
    args=parser.parse_args(argv)
    if args.command=="dashboard":
        if not 1024<=args.port<=65535:
            parser.error("Dashboard port must be unprivileged")
        return subprocess.call([sys.executable,"-m","streamlit","run",
                                str(ROOT/"src/ucl/dashboard.py"),
                                "--server.address","127.0.0.1","--server.port",str(args.port),
                                "--server.headless","true","--browser.gatherUsageStats","false"],
                               cwd=ROOT)
    if args.command=="matrix":
        configurations=[f"EXP-{i:03d}" for i in range(1,6)]+[
            "CONTROL-RIM-MAX","CONTROL-GROWTH-NONE","CONTROL-GROWTH-PLATEAU"]
        if args.include_llm:
            configurations += ["EXP-006","EXP-007","EXP-008"]
        for seed in map(int,args.seeds.split(",")):
            for config in configurations:
                subprocess.run([sys.executable,"-B","-m","ucl.cli","train",
                                "--config",f"configs/{config}.json","--seed",str(seed)],
                               cwd=ROOT,check=True)
        return 0
    if args.command=="study-pilot":
        return study_pilot(args,parser)
    if args.command=="study-build":
        from .study_panels import build_study
        result=build_study(ROOT/STUDY_DIR/"protocol.json",ROOT/STUDY_DIR,force=args.force)
    elif args.command=="study-run":
        result=study_run(args,parser)
    elif args.command=="study-record":
        from .study_recording import record_study_episodes
        output=Path(args.output)
        result=record_study_episodes(ROOT,args.run,args.checkpoint,args.task,args.panel,
                                     split_list(args.maps),
                                     output if output.is_absolute() else ROOT/output)
    elif args.command=="study-reevaluate":
        from .study_reevaluation import reevaluate_run
        reevaluated=reevaluate_run(ROOT,args.run,args.replicas)
        result={key:reevaluated[key]
                for key in ("status","source_run","replicas","elapsed_seconds")}
    elif args.command=="study-chain":
        from .study_chain import chain_status, load_plan, run_chain
        plan_path=Path(args.plan)
        plan=load_plan(plan_path if plan_path.is_absolute() else ROOT/plan_path)
        result=chain_status(ROOT,plan) if args.status else run_chain(
            ROOT,plan,retry=args.retry,limit=args.limit)
    elif args.command=="study-variance":
        from .study_chain import load_plan
        from .study_variance import analyze_plan, brief
        plan_path=Path(args.plan)
        plan=load_plan(plan_path if plan_path.is_absolute() else ROOT/plan_path)
        result=brief(analyze_plan(ROOT,plan,args.bootstrap))
    elif args.command=="study-sealed-build":
        from .study_sealed import build_sealed_panel
        built=build_sealed_panel(ROOT,args.study,args.task,args.base,args.size,args.name)
        result={key:built[key] for key in ("name","task_id","base","scanned","excluded_maps")}
        result["maps"]=len(built["seeds"])
    elif args.command=="study-sealed-open":
        from .study_sealed import open_sealed
        result=open_sealed(ROOT,args.spec)
    elif args.command=="study-probe-time":
        from .study_probe import probe_plan
        result=probe_plan(ROOT,args.plan,split_list(args.labels),args.node,args.checkpoint,
                          args.replicas,args.id)["summary"]
    elif args.command=="study-contrasts":
        from .study_contrasts import analyze_spec, brief
        result=brief(analyze_spec(ROOT,args.spec))
    elif args.command=="compare":
        from .reports import comparison_report
        result=comparison_report(args.runs,Path(args.output))
    elif args.command=="learning-to-learn":
        from .reports import run_learning_to_learn
        result=run_learning_to_learn(read_config(args.config),args.seed,args.output)
    else:
        from .runner import run_experiment
        config=read_config(args.config)
        if args.seed is not None:
            config["seed"]=args.seed
        if args.steps is not None:
            if args.steps < 0:
                parser.error("--steps cannot be negative")
            config["train_steps"]=args.steps
        mode="evaluate" if args.command=="evaluate" else "train"
        result=run_experiment(config,args.output,mode=mode,checkpoint=args.checkpoint,
                              task_id=args.task,curriculum=args.curriculum)
    print(json.dumps(result,indent=2,allow_nan=False,default=str))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
