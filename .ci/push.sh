#!/usr/bin/env bash
# Submit the current commit: build the agent image, push it to ECR, and upload
# artifacts/outputs to S3. Run from the repo root with ./.ci/push.sh
#
# Requires AWS credentials in the environment (or ~/.aws) plus:
#   AWS_REGION          e.g. eu-central-1
#   ECR_REPOSITORY_URI  e.g. 123456789012.dkr.ecr.eu-central-1.amazonaws.com/sphere-forge/<team>
#   S3_SUBMISSION_URI   e.g. s3://sphere-forge-submissions/<team>
set -euo pipefail

: "${AWS_REGION:?must be set}"
: "${ECR_REPOSITORY_URI:?must be set}"
: "${S3_SUBMISSION_URI:?must be set}"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit your changes before submitting, so the image and outputs match a commit." >&2
    exit 1
fi

if ! compgen -G "artifacts/outputs/test_case_*.stp" >/dev/null; then
    echo "No artifacts/outputs/test_case_*.stp to submit." >&2
    exit 1
fi

COMMIT=$(git rev-parse HEAD)
IMAGE="$ECR_REPOSITORY_URI:$COMMIT"
SUBMISSION="$S3_SUBMISSION_URI/$COMMIT"

aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${ECR_REPOSITORY_URI%%/*}"

docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"

aws s3 cp artifacts/outputs "$SUBMISSION/outputs" --recursive --exclude "*" --include "test_case_*.stp"

# Written last, so the eval pipeline can treat it as the "submission complete" marker
printf '{"commit": "%s", "image": "%s"}\n' "$COMMIT" "$IMAGE" \
    | aws s3 cp - "$SUBMISSION/submission.json"

echo "Submitted $COMMIT"
