#!/usr/bin/env python3
"""
Run a SageMaker Batch Transform job using an existing SageMaker *Model* name.

This does NOT train; it assumes a model already exists in SageMaker (created from training output).
Best for low cost demos vs. keeping an endpoint running.

You will run this from SageMaker Studio OR locally if your AWS creds are set.

Usage:
  python scripts/module5_batch_transform.py \
    --region us-west-2 \
    --model-name <existing_sagemaker_model_name> \
    --input-s3 s3://<bucket>/<prefix>/batch_input/ \
    --output-s3 s3://<bucket>/<prefix>/batch_output/ \
    --job-name aai540-g2-batch-001
"""

import argparse
import time

import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--input-s3", required=True)
    ap.add_argument("--output-s3", required=True)
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--instance-count", type=int, default=1)
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

    resp = sm.create_transform_job(
        TransformJobName=args.job_name,
        ModelName=args.model_name,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": args.input_s3}},
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={"S3OutputPath": args.output_s3, "AssembleWith": "Line"},
        TransformResources={
            "InstanceType": args.instance_type,
            "InstanceCount": args.instance_count,
        },
    )
    print("Started transform job:", args.job_name)

    # Wait for completion (simple poll)
    while True:
        d = sm.describe_transform_job(TransformJobName=args.job_name)
        status = d["TransformJobStatus"]
        print("Status:", status)
        if status in ("Completed", "Failed", "Stopped"):
            print("Final status:", status)
            if status != "Completed":
                print("Failure reason:", d.get("FailureReason"))
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
