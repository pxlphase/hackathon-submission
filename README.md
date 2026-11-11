# Sphere Forge \#1: Prompt to CAD

Welcome to Sphere's first hackathon! Can AI generate production-ready 3D CAD models directly from text prompts or images? We're putting it to the test at Sphere Forge #1!

Your challenge: Build an autonomous agent capable of turning natural language inputs into precise 3D geometry. Submissions will be benchmarked on geometric accuracy against a hidden target component. The team with the highest similarity score takes home the top prize.

# Setup

Each team works in its own **private** repository, created from this template under one of your
GitHub accounts (or your own GitHub organisation). Organisers don't need access to it: submissions
are made from your CI to our container registry and bucket, using credentials we give you.

1. Create your repo from this template, and keep it private:

   ```sh
   gh repo create <your-account>/sphere-forge-<team> --private --clone \
       --template sphere-energy/hackathon-submission
   ```

   or use **Use this template → Create a new repository** on GitHub.
2. Add your teammates under **Settings → Collaborators**.
3. The organisers will send you two files: `<team>.vars.env` and `<team>.secrets.env`. From inside
   your repo, load them into GitHub Actions. Do this as the repo's owner: on a personal account,
   collaborators can't manage secrets.

   ```sh
   gh variable set -f <team>.vars.env
   gh secret set -f <team>.secrets.env
   ```

   Then delete `<team>.secrets.env`, and never commit it. Until these are set, CI runs tests but
   skips submitting.

CI runs on your account's GitHub Actions minutes (2,000/month free for private repos on a personal
account).

# Repo overview

# Submission format

Your submission comprises of two parts: i) docker image containing your agent workflow, and ii) your output CAD .stp files. The docker image should be built from the root of the repo, and should contain all dependencies required to run your agent workflow. The output .stp files should be placed in the `artifacts/outputs` directory.

The `artifacts` structure should be:

```
artifacts
├── inputs
│   ├── test_case_1.txt
│   ├── test_case_1.jpg
│   ├── test_case_2.txt
│   ├── test_case_2.jpg
│   └── ...
├── outputs
│   ├── test_case_1.stp
│   ├── test_case_2.stp
│   └── ...
```

Each test case consists of a text prompt and an image input. Your agent workflow should process each test case and generate a corresponding .stp file in the `outputs` directory. The naming convention is `test_case_<n>.stp` where `<n>` is the test case number. <b><u>This naming convention is strict</u></b>. You should commit the output files (overwrite).

We have provided [artifacts/example_outputs](./artifacts/example_outputs) as reference for the expected output format.

# Submission flow

The repo is set up to faciliate the submission of your agent workflow with the inclusion of some CI boilerplate. The recommended submission flow is as follows:

1. Create a development branch for a set of changes to your agent workflow, e.g. `feat-agent-orchestration`
2. Make your changes and test them locally (simplest case run directly on VM)
3. Create PR to merge your changes into the `main` branch
4. CI on PR branch will run tests and build the docker image to ensure your changes are valid
5. Upon successful merge to `main` (once your repo is configured, see [Setup](#setup)), CI will i) rebuild the docker image, ii) push the image to the submission container registry (ECR), and iii) upload your committed output .stp files to the submission bucket (S3). Each submission is tied to its commit and can't be overwritten, so re-running CI for an already-submitted commit fails; push a new commit instead.
6. Our eval pipeline will detect the new submission and run the evaluation automatically, updating the leaderboard with your score.

You are free to adjust the workflow to your needs, i.e. using `dev` branches and only merging stable versions to `main` etc. The current CI boilerplate is setup to push built images to the submission registry upon each successful merge to the `main` branch. You can also submit manually from the root of the repo, with the organisers' files loaded into your environment:

```sh
set -a; source <team>.vars.env; source <team>.secrets.env; set +a
./.ci/push.sh
```

<u><b>Ensure you do not alter the submission format however</b></u>. Submissions in unexpected formats will fail to be evaluated and will not be scored.

# Runtime limits

Your pipeline is run under these limits during evaluation:

| Limit | Value |
| --- | --- |
| Wall-clock time | **10 minutes** end to end, for the full run over all test cases |
| GPU | One NVIDIA RTX PRO 6000 |
| VRAM | **96 GB** in total. Everything your pipeline runs on the GPU (models, KV cache, CAD tooling) must fit in this |

Test your pipeline end to end on a single RTX PRO 6000 before you submit, and leave some headroom
on both time and memory.

# VM Access

VMs are provided through [NVIDIA Brev](https://brev.nvidia.com). For anything not covered here,
see the [Brev docs](https://docs.nvidia.com/brev/latest/).

## Install and log in

Follow [CLI getting started](https://docs.nvidia.com/brev/latest/cli/getting-started). In short:

```sh
# macOS
brew install brevdev/homebrew-brev/brev

# Linux / WSL (don't run with sudo)
bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"

brev --version
brev login            # opens a browser to sign in
```

## Connect to your instance

See [CLI connectivity](https://docs.nvidia.com/brev/latest/cli/connectivity) and
[instance management](https://docs.nvidia.com/brev/latest/cli/instance-management).

```sh
brev list                       # list instances in your org
brev shell <instance>           # open a shell (alias: brev ssh)
brev open <instance>            # open in VS Code (or: brev open <instance> cursor)
brev exec <instance> "nvidia-smi"
```

`brev` manages your SSH config for you, so a plain `ssh <instance>` also works. Run `brev refresh`
if you created the instance in the web console, or if you restarted it (its IP may have changed).

## Hosting an LLM and tunnelling to it

You can serve a model on your Brev instance (for example with [vLLM](https://docs.vllm.ai)) and
reach it from your laptop through an SSH tunnel:

```sh
# On the instance: serve an OpenAI-compatible API on port 8000
vllm serve <model> --port 8000 --api-key <your-key>

# On your laptop: forward local port 8000 to port 8000 on the instance
brev port-forward <instance> --port 8000:8000
# or with plain ssh:
ssh -N -L 8000:localhost:8000 <instance>
```

The endpoint is now at `http://localhost:8000/v1` on your laptop. If you run your agent in a local
Docker container, `localhost` points at the container itself, not your laptop. Use
`http://host.docker.internal:8000/v1` instead. On Linux, also pass
`--add-host=host.docker.internal:host-gateway` to `docker run`.

`brev port-forward` takes one port mapping per command. Run it once for each port you need.

## LLM endpoint during evaluation

The eval passes your container an LLM endpoint as two environment variables:

- `LLM_BASE_URL`: an OpenAI-compatible base URL, for example a gateway you run, a hosted
  OpenAI-compatible API, or vLLM on your eval box.
- `LLM_API_KEY`: the key for that endpoint.

Read both from the environment in your agent. Don't hardcode `localhost` or your own keys.

**You must keep your own endpoint up and reachable until judging ends.** If it goes down, your
evaluation runs will fail.


# Evaluation

The eval pipeline is hidden from the participants, and will run automatically in the background upon each new submission bundle, i.e. each new set of docker image + .stp files pushed to the submission registries. Scores are computed and stored in the leaderboard, which will be visible to everyone. Race other teams to the top by submitting new pipelines and improving your agent's performance!