# Cost Guardrails (AAI-540 Final Project)

Goal: keep AWS Learner Lab spend low while still demonstrating required MLOps components.

## Rules we follow
- Prefer **local feature extraction** (CSV/Parquet artifacts) and only upload the final datasets to S3.
- Use **small instance types** for training/inference whenever possible.
- **Delete endpoints immediately after testing** (endpoints burn credits continuously).
- Use **Batch Transform** for demos when possible (runs then stops).
- Keep datasets and logs under a single S3 prefix so cleanup is easy.

## Required cleanup
After every AWS run:
- Delete SageMaker endpoint + endpoint config
- Delete SageMaker model
- Stop running training/transform jobs
- Remove S3 objects under project prefix if needed

## Cleanup script
Use:
- Dry run:
  `python scripts/aws_cleanup.py --region <region> --project <project>`

- Actually delete:
  `python scripts/aws_cleanup.py --region <region> --project <project> --confirm`
