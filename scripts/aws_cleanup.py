#!/usr/bin/env python3
"""
Safe cleanup utility for the AAI-540 project.

Default behavior is DRY RUN (no deletes). To actually delete, pass --confirm.

What it cleans (by project prefix/name match):
- SageMaker Endpoints + Endpoint Configs
- SageMaker Models
- SageMaker Training Jobs (stops if InProgress)
- Batch Transform Jobs
- (Optional) Model Package Groups / Model Packages (if you enable it)
- S3 objects under a prefix (optional)

Usage examples:
  python scripts/aws_cleanup.py --region us-west-2 --project aai540-g2 --dry-run
  python scripts/aws_cleanup.py --region us-west-2 --project aai540-g2 --confirm
  python scripts/aws_cleanup.py --region us-west-2 --project aai540-g2 --confirm --s3-bucket <bucket> --s3-prefix <prefix>
"""

import argparse
import sys
import time
from typing import List, Optional

import boto3


def _matches(name: str, project: str) -> bool:
    n = name.lower()
    p = project.lower()
    return p in n


def _log(msg: str):
    print(msg, flush=True)


def _safe_call(dry_run: bool, action_desc: str, fn, *args, **kwargs):
    if dry_run:
        _log(f"[DRY_RUN] {action_desc}")
        return None
    _log(f"[EXEC] {action_desc}")
    return fn(*args, **kwargs)


def list_all(sm, list_fn_name: str, result_key: str, **kwargs) -> List[dict]:
    out = []
    token = None
    list_fn = getattr(sm, list_fn_name)
    while True:
        if token:
            kwargs["NextToken"] = token
        resp = list_fn(**kwargs)
        out.extend(resp.get(result_key, []))
        token = resp.get("NextToken")
        if not token:
            break
    return out


def cleanup_sagemaker(sm, project: str, dry_run: bool):
    # 1) Endpoints
    endpoints = list_all(sm, "list_endpoints", "Endpoints", SortBy="CreationTime", SortOrder="Descending")
    for ep in endpoints:
        name = ep["EndpointName"]
        if _matches(name, project):
            _safe_call(dry_run, f"Delete endpoint: {name}", sm.delete_endpoint, EndpointName=name)

    # 2) Endpoint configs
    ep_configs = list_all(sm, "list_endpoint_configs", "EndpointConfigs", SortBy="CreationTime", SortOrder="Descending")
    for cfg in ep_configs:
        name = cfg["EndpointConfigName"]
        if _matches(name, project):
            _safe_call(dry_run, f"Delete endpoint config: {name}", sm.delete_endpoint_config, EndpointConfigName=name)

    # 3) Models
    models = list_all(sm, "list_models", "Models", SortBy="CreationTime", SortOrder="Descending")
    for m in models:
        name = m["ModelName"]
        if _matches(name, project):
            _safe_call(dry_run, f"Delete model: {name}", sm.delete_model, ModelName=name)

    # 4) Transform jobs
    t_jobs = list_all(sm, "list_transform_jobs", "TransformJobSummaries", SortBy="CreationTime", SortOrder="Descending")
    for tj in t_jobs:
        name = tj["TransformJobName"]
        if _matches(name, project):
            # no delete API; stop if running
            desc = sm.describe_transform_job(TransformJobName=name)
            status = desc.get("TransformJobStatus")
            if status in ("InProgress", "Stopping"):
                _safe_call(dry_run, f"Stop transform job: {name}", sm.stop_transform_job, TransformJobName=name)
            else:
                _log(f"[INFO] Transform job kept (completed/failed): {name} ({status})")

    # 5) Training jobs
    tr_jobs = list_all(sm, "list_training_jobs", "TrainingJobSummaries", SortBy="CreationTime", SortOrder="Descending")
    for tj in tr_jobs:
        name = tj["TrainingJobName"]
        if _matches(name, project):
            desc = sm.describe_training_job(TrainingJobName=name)
            status = desc.get("TrainingJobStatus")
            if status in ("InProgress", "Stopping"):
                _safe_call(dry_run, f"Stop training job: {name}", sm.stop_training_job, TrainingJobName=name)
            else:
                _log(f"[INFO] Training job kept (completed/failed): {name} ({status})")

    _log("[DONE] SageMaker cleanup pass finished.")


def cleanup_s3(s3, bucket: str, prefix: str, dry_run: bool):
    _log(f"[INFO] S3 cleanup bucket={bucket} prefix={prefix}")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            keys.append({"Key": obj["Key"]})

    if not keys:
        _log("[INFO] No S3 objects found for prefix.")
        return

    _log(f"[INFO] Found {len(keys)} objects under s3://{bucket}/{prefix}")
    # Delete in chunks of 1000
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        _safe_call(
            dry_run,
            f"Delete S3 objects [{i}-{i+len(chunk)-1}] under s3://{bucket}/{prefix}",
            s3.delete_objects,
            Bucket=bucket,
            Delete={"Objects": chunk, "Quiet": True},
        )
    _log("[DONE] S3 cleanup finished.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="AWS region (e.g., us-west-2)")
    ap.add_argument("--project", required=True, help="Project identifier substring used in AWS resource names")
    ap.add_argument("--confirm", action="store_true", help="Actually delete resources (otherwise dry run)")
    ap.add_argument("--s3-bucket", default=None, help="Optional: S3 bucket to clean")
    ap.add_argument("--s3-prefix", default=None, help="Optional: S3 prefix to clean")
    args = ap.parse_args()

    dry_run = not args.confirm
    _log(f"[MODE] {'DRY_RUN' if dry_run else 'DESTRUCTIVE'}")

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    cleanup_sagemaker(sm, args.project, dry_run)

    if args.s3_bucket and args.s3_prefix:
        cleanup_s3(s3, args.s3_bucket, args.s3_prefix, dry_run)

    _log("[OK] Cleanup script completed.")


if __name__ == "__main__":
    main()
