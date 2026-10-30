# Sphere Forge \#1: Prompt to CAD

Welcome to Sphere's first hackathon! Can AI generate production-ready 3D CAD models directly from text prompts or images? We're putting it to the test at Sphere Forge #1!

Your challenge: Build an autonomous agent capable of turning natural language inputs into precise 3D geometry. Submissions will be benchmarked on geometric accuracy against a hidden target component. The team with the highest similarity score takes home the top prize.

# Setup

# Repo overview

# Submission format

Your submission comprises of two parts: i) docker image containing your agent workflow, and ii) your output CAD .stp files. The docker image should be built from the root of the repo, and should contain all dependencies required to run your agent workflow. The output .stp files should be placed in the `output` directory at the root of the repo.

The `artifacts` structure should be:

```
artifacts
├── inputs
│   ├── test_case_1.txt
│   ├── test_case_1.png
│   ├── test_case_2.txt
│   ├── test_case_2.png
│   └── ...
├── outputs
│   ├── test_case_1.stp
│   ├── test_case_2.stp
│   └── ...
```

Each test case consists of a text prompt and an image input. Your agent workflow should process each test case and generate a corresponding .stp file in the `outputs` directory. The naming convention is `test_case_<n>.stp` where `<n>` is the test case number. <b><u>This naming convention is strict</b></u>. You should commit the output files (overwrite).

We have provided [artifacts/example_outputs](./artifacts/example_outputs) as reference for the expected output format.

# Submission flow

The repo is set up to faciliate the submission of your agent workflow with the inclusion of some CI boilerplate. The recommended submission flow is as follows:

1. Create a development branch for a set of changes to your agent workflow, e.g. `feat-agent-orchestration`
2. Make your changes and test them locally (simplest case run directly on VM)
3. Create PR to merge your changes into the `main` branch
4. CI on PR branch will run tests and build the docker image to ensure your changes are valid
5. Upon successful merge to `main`, CI will i) rebuild the docker image, ii) push the image to the submission container registry (ECR), and iii) upload your committed output .stp files to the submission bucket (S3). 
6. Our eval pipeline will detect the new submission and run the evaluation automatically, updating the leaderboard with your score.

You are free to adjust the workflow to your needs, i.e. using `dev` branches and only merging stable versions to `main` etc. The current CI boilerplate is setup to push built images to the submission registry upon each successful merge to the `main` branch. You can also push images to the submission registry manually by running `source ./.ci/push.sh` from the root of the repo.

<u><b>Ensure you do not alter the submission format however</b></u>. Submissions in unexpected formats will fail to be evaluated and will not be scored.

# VM Access

# Evaluation

The eval pipeline is hidden from the participants, and will run automatically in the background upon each new submission bundle, i.e. each new set of docker image + .stp files pushed to the submission registries. Scores are computed and stored in the leaderboard, which will be visible to everyone. Race other teams to the top by submitting new pipelines and improving your agent's performance!