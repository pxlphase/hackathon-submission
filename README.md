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
│   ├── test_case_1
│   │   ├── description.md
│   │   ├── context.png          (only for some test cases, see below)
│   │   ├── facts.json
│   │   └── images
│   │       ├── view_0_front.png
│   │       ├── ...
│   │       └── view_7_iso_back_left_bottom.png
│   ├── test_case_2
│   └── ...                      (9 test cases)
├── outputs
│   ├── test_case_1.stp
│   ├── test_case_2.stp
│   └── ...
```

## Inputs

Each test case is one mechanical part, given as a written design brief and rendered views:

| file | contents |
| --- | --- |
| `description.md` | design brief: function, envelope, features with exact dimensions in mm |
| `images/view_*.png` | 8 orthographic views of the part, 1024 px |
| `context.png` | the part highlighted in orange inside its assembly. **Optional**: only present for parts taken from an assembly, so your pipeline must work without it |
| `facts.json` | measurements the brief was written from: bounding box, volume, hole/fillet inventory, ... |

All views are orthographic and rendered in the part's own X/Y/Z frame:

| file | camera |
| --- | --- |
| `view_0_front` | looking along +Y: X to the right, Z up |
| `view_1_back` | looking along -Y: X to the left, Z up |
| `view_2_left` | looking along +X: Y to the left, Z up |
| `view_3_right` | looking along -X: Y to the right, Z up |
| `view_4_top` | looking along -Z: X to the right, Y up |
| `view_5_bottom` | looking along +Z: X to the right, Y down |
| `view_6_iso_front_right_top` | isometric from +X, -Y, +Z |
| `view_7_iso_back_left_bottom` | isometric from -X, +Y, -Z |

Positions in `description.md` are in mm from the minimum corner of the part's bounding box.

The inputs committed here are the **practice set**: the leaderboard scores your outputs for these
parts live during the hackathon. The final ranking re-runs the top teams' images on a **hidden
set** of 9 different parts, in exactly this format (3 simple, 3 medium, 3 complex). Build for
parts you haven't seen, not for these 9.

## Outputs

Your agent workflow should process each test case and generate a corresponding .stp file in the
`outputs` directory: one solid per test case, in millimetres. The naming convention is
`test_case_<n>.stp` where `<n>` is the test case number. <b><u>This naming convention is
strict</u></b>, including the `.stp` extension. A missing, broken or non-solid file scores 0 for
that test case. Position and orientation do not matter; size and handedness do. You should commit
the output files (overwrite).

Your image is run as `python scripts/main.py --input_dir <inputs> --output_dir <outputs>`, with
`<inputs>` laid out like `artifacts/inputs`. Keep that entrypoint and those arguments.

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

Your image is run under these limits when we re-run it after code freeze
([Code-freeze handover](#code-freeze-handover)):

| Limit | Value |
| --- | --- |
| Wall-clock time | **20 minutes** end to end, for the full run over all test cases |
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

The tunnel is for development only. When we run your image after code freeze there is no tunnel:
see [Code-freeze handover](#code-freeze-handover).

# Configuration

Your agent's configuration lives in two places:

- **Everything that defines your agent goes into the image**: model name, prompts, temperatures,
  tool settings, in code, config files or `ENV` lines in the [Dockerfile](Dockerfile). It's frozen
  with your commit, so the image we re-run after code freeze is the agent that produced your
  submitted outputs.
- **Read the LLM endpoint from `OPENAI_BASE_URL` and `OPENAI_API_KEY`.** The OpenAI Python SDK
  picks both up by itself (`OpenAI()`), and most other clients take them as arguments:

  ```python
  import os
  from openai import OpenAI

  client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY
  model = os.environ.get("OPENAI_MODEL", "<your model>")
  ```

  Set your development values as `ENV` in the Dockerfile (there are commented lines to start
  from), so the image you submit runs as it is. Other variables your agent reads, e.g. a second
  endpoint, work the same way; just list them in your handover file below.

A local `.env` is fine for development: copy [.env.example](.env.example). It's ignored by git
and by `docker build`, so nothing in the image may depend on it. Whatever you put in the
Dockerfile ends up in your image in our registry, which only your team and the organisers can
pull: the key for your own vLLM endpoint is fine there, a personal provider key isn't.

## Code-freeze handover

After code freeze we re-run the top teams' images on our own machine, with your handed-over
environment, as a non-root user, killed after 20 minutes:

```sh
docker run --env-file <team>.env -v <inputs>:/data/inputs:ro -v <outputs>:/data/outputs <your image> \
    python scripts/main.py --input_dir /data/inputs --output_dir /data/outputs
```

1. **On the practice inputs**: the files it writes must match the outputs you submitted for that
   commit. A mismatch doesn't change your score by itself, but we review it by hand.
2. **On the hidden inputs**: these outputs decide the final ranking.

So at code freeze, **send the organisers `<team>.env`**: every environment variable your agent
reads, with the values for these runs. One `KEY=VALUE` per line, without quotes or `export`:

```sh
OPENAI_BASE_URL=http://<your Brev instance's public address>:8000/v1
OPENAI_API_KEY=<the --api-key your vLLM serves with>
OPENAI_MODEL=<model>      # and anything else your agent reads
```

It's passed to your container on top of the `ENV` in your image, so its values win. We may edit
it to keep your runs working, e.g. if your instance's address changes, and we'll tell you if we
do; tell us if anything changes after you've sent it. Your endpoint must:

- **be reachable from the internet, without a tunnel.** `localhost` and `host.docker.internal`
  only work on your laptop. Expose the port through Brev (see the
  [Brev docs](https://docs.nvidia.com/brev/latest/)) and keep `--api-key` on.
- **serve the same model, with the same settings, as when you made your submitted outputs**, and
  stay up until judging ends. If it's down or has changed, your runs fail or stop matching.

Check both before you hand over, from a machine that isn't tunnelled to your instance, by
running your image exactly as we will:

```sh
curl -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models"   # with <team>.env loaded
docker build --platform linux/amd64 -t my-agent .
mkdir -p out && docker run --rm --user "$(id -u):$(id -g)" --env-file <team>.env \
    -v "$PWD/artifacts/inputs:/data/inputs:ro" -v "$PWD/out:/data/outputs" \
    my-agent python scripts/main.py --input_dir /data/inputs --output_dir /data/outputs
```

`out/` should then match your committed `artifacts/outputs/`. Write only to the output directory
and `/tmp`: the container doesn't run as root.


# Evaluation

The eval pipeline is hidden from the participants, and will run automatically in the background upon each new submission bundle, i.e. each new set of docker image + .stp files pushed to the submission registries. Scores are computed and stored in the leaderboard, which will be visible to everyone. Race other teams to the top by submitting new pipelines and improving your agent's performance!