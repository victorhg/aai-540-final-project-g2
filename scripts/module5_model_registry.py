#!/usr/bin/env python3
"""
Register an existing trained model artifact into SageMaker Model Registry + create a Model Card.

This is meant to run in SageMaker Studio (recommended), but works locally if AWS creds allow.

You must provide:
- model_package_group_name
- model_data_url (S3 path to model.tar.gz)
- image_uri (the inference container image used)

Usage:
  python scripts/module5_model_registry.py \
    --region us-west-2 \
    --group-name aai540-g2-ser \
    --model-data s3://<bucket>/<prefix>/output/model.tar.gz \
    --image-uri <account>.dkr.ecr.<region>.amazonaws.com/<image>:<tag> \
    --model-metrics-s3 s3://<bucket>/<prefix>/metrics.json
"""

import argparse
import json
import time

import boto3


def ensure_group(sm, group_name: str, desc: str):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group_name)
        print("[OK] Model Package Group exists:", group_name)
        return
    except sm.exceptions.ResourceNotFound:
        pass

    sm.create_model_package_group(
        ModelPackageGroupName=group_name,
        ModelPackageGroupDescription=desc[:250],
    )
    print("[CREATED] Model Package Group:", group_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--group-name", required=True)
    ap.add_argument("--model-data", required=True, help="S3 URI to model.tar.gz")
    ap.add_argument("--image-uri", required=True)
    ap.add_argument("--model-metrics-s3", default=None, help="Optional S3 URI to a metrics json")
    ap.add_argument("--model-approval", default="PendingManualApproval")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

    ensure_group(
        sm,
        args.group_name,
        desc="Speech Emotion Recognition (SER) models for AAI-540 final project (CREMA-D + multilingual extensions).",
    )

    inference_spec = {
        "Containers": [
            {
                "Image": args.image_uri,
                "ModelDataUrl": args.model_data,
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
    }

    # Optional metrics block (links to S3)
    model_metrics = None
    if args.model_metrics_s3:
        model_metrics = {
            "ModelQuality": {
                "Statistics": {"ContentType": "application/json", "S3Uri": args.model_metrics_s3}
            }
        }

    package_name = f"{args.group_name}-pkg-{int(time.time())}"

    create_args = dict(
        ModelPackageGroupName=args.group_name,
        ModelPackageDescription="Experiment package: feature-based classifier for SER (Module 4/5 iteration).",
        InferenceSpecification=inference_spec,
        ModelApprovalStatus=args.model_approval,
        ModelPackageName=package_name,
    )
    if model_metrics:
        create_args["ModelMetrics"] = model_metrics

    resp = sm.create_model_package(**create_args)
    arn = resp["ModelPackageArn"]
    print("[CREATED] Model Package:", arn)

    # Create a Model Card (basic but complete)
    card_name = f"{args.group_name}-card-{int(time.time())}"
    card = {
        "model_overview": {
            "model_name": "SER Feature Classifier",
            "model_description": "Speech Emotion Recognition using engineered acoustic features (MFCC + spectral + prosodic).",
            "problem_type": "Multi-class classification",
            "model_owner": "Group 2",
        },
        "intended_uses": {
            "intended_uses": ["Academic project demonstration of MLOps pipeline (AAI-540)."],
            "factors": ["Speaker identity, language, recording conditions can affect performance."],
            "out_of_scope_uses": ["High-stakes decisions about individuals."],
        },
        "training_details": {
            "training_observations": "Model trained on processed feature tables derived from audio files; uses train/test split.",
            "training_data": "CREMA-D features (+ planned multilingual SER datasets).",
            "hyperparameters": "See training job / notebook logs; this registry package captures the deployed artifact.",
        },
        "evaluation_details": {
            "metrics": ["accuracy", "weighted_f1"],
            "results_location": "See artifacts in repo and optional metrics JSON linked in Model Package.",
        },
        "ethical_considerations": {
            "ethical_considerations": [
                "Emotion recognition can be sensitive; results may be biased by language/culture/speaker demographics.",
                "Use only for demonstration and research, not for punitive or discriminatory outcomes.",
            ]
        },
        "limitations": {
            "known_limitations": [
                "Performance depends heavily on feature quality and dataset distribution.",
                "Generalization across languages may be limited without multilingual training data.",
            ]
        },
    }

    sm.create_model_card(
        ModelCardName=card_name,
        ModelCardStatus="Draft",
        Content=json.dumps(card),
    )
    print("[CREATED] Model Card:", card_name)

    # Print describe outputs for screenshotting later
    print("\n--- describe_model_package_group ---")
    print(sm.describe_model_package_group(ModelPackageGroupName=args.group_name))

    print("\n--- describe_model_package ---")
    print(sm.describe_model_package(ModelPackageName=package_name))

    print("\n--- describe_model_card ---")
    print(sm.describe_model_card(ModelCardName=card_name))


if __name__ == "__main__":
    main()
