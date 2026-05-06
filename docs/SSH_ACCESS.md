# SSH access to the H100 instance

Quick onboarding for the second operator (Maryna). Assumes you've been sent `secrets.local.sh` over a secure channel (Signal, 1Password, etc.) — **never** through GitHub, email, or Slack.

## Prerequisites on your laptop

- macOS or Linux with a modern OpenSSH client (`ssh -V` should print at least 9.x).
- AWS CLI v2 (`aws --version`) configured with credentials that can at minimum:
  - `ec2:DescribeInstances` (to look up the public IP if it changes)
  - `s3:Get/PutObject` on `s3://${S3_BUCKET}/*` (to follow the runbook's evacuation flow)
  - The team's IAM admin can either share `paper-drochila`'s creds or grant you a separate user.

## Step 1 — load the run-specific values

Save the file you were sent as `secrets.local.sh` somewhere outside the repo (it's gitignored anyway). Then in every new shell where you'll run AWS or SSH:

```bash
source ~/path/to/secrets.local.sh
echo "$S3_BUCKET $SUBNET_ID $KEY_NAME"   # quick sanity print
```

This sets `AWS_REGION`, `S3_BUCKET`, the resource IDs, and `KEY_NAME=sralnik-h100`.

## Step 2 — get an SSH private key

Only one private key (`~/.ssh/sralnik-h100.pem`) was generated when the instance was launched. EC2 only accepts SSH for users whose public key is in the instance's `~ubuntu/.ssh/authorized_keys`. Two ways to get in:

### Option A — receive Aleksandr's `.pem` (fastest, fine for short runs)

1. Aleksandr sends you `sralnik-h100.pem` via 1Password / Signal (encrypted). **Never email or Slack the .pem.**
2. Save it locally and lock its permissions:
   ```bash
   mv ~/Downloads/sralnik-h100.pem ~/.ssh/sralnik-h100.pem
   chmod 400 ~/.ssh/sralnik-h100.pem
   ```

### Option B — add your own public key (better hygiene)

1. Generate (skip if you already have one):
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/sralnik_maryna -C "maryna@$(hostname)"
   ```
2. Send `~/.ssh/sralnik_maryna.pub` (the **public** half) to Aleksandr.
3. Aleksandr appends it to the instance's authorized_keys:
   ```bash
   # On Aleksandr's laptop:
   ssh sralnik 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/sralnik_maryna.pub
   ```
4. You then use `~/.ssh/sralnik_maryna` as your `IdentityFile` in step 3 below (instead of `~/.ssh/sralnik-h100.pem`).

## Step 3 — `~/.ssh/config` entry

Find the current public IP. The instance's IP can change if it gets relaunched (Capacity Block lets you stop/start within the window), so don't hardcode it:

```bash
source ~/path/to/secrets.local.sh
PUB_IP=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:Name,Values=sralnik-h100-24h" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text)
echo "Public IP: $PUB_IP"
```

Append to `~/.ssh/config` (create the file if it doesn't exist; `chmod 600 ~/.ssh/config` if new):

```
Host sralnik
  HostName <paste-the-PUB_IP-here>
  User ubuntu
  IdentityFile ~/.ssh/sralnik-h100.pem      # or ~/.ssh/sralnik_maryna under Option B
  ServerAliveInterval 60
  ServerAliveCountMax 10
```

If you'd rather not edit the file in two places when the IP changes, use a `Match exec` block (advanced) or just re-paste the new IP each time.

## Step 4 — connect

```bash
ssh sralnik
```

First time, `ssh` asks you to accept the instance's host key — type `yes`. You should land on `ubuntu@ip-172-31-96-NNN`.

Smoke test:
```bash
ssh sralnik 'hostname && nvidia-smi -L'
```
You should see 8 lines like `GPU 0: NVIDIA H100 80GB HBM3 ...`.

## Step 5 — VSCode Remote-SSH (optional but recommended)

1. Install the `ms-vscode-remote.remote-ssh` extension on your laptop's VSCode.
2. `Cmd+Shift+P` → **Remote-SSH: Connect to Host…** → `sralnik` (it picks up the SSH config).
3. **File → Open Folder…** → `/mnt/data/sralnik/repo`.
4. In the integrated terminal, run extensions on the remote:
   ```bash
   for ext in anthropic.claude-code ms-python.python ms-toolsai.jupyter charliermarsh.ruff eamodio.gitlens; do
     code --install-extension "$ext"
   done
   ```
5. Use VSCode for editing/notebooks; keep training sessions in **regular SSH + tmux**, not VSCode terminals (those die on connection drops).

## Step 6 — what to do when you're on the instance

- Training session: `tmux attach -t sralnik` (Aleksandr's main session).
- Detach without killing: `Ctrl+b d`.
- Pop a new pane to run helper commands without disturbing training: `Ctrl+b "` (split horizontal) or `Ctrl+b %` (split vertical).
- Refer to `docs/RUN_PLAN.md` §"Debugging on the instance" before making any code edits — there's a separate `git worktree` at `/mnt/data/sralnik/debug` for triage so the running training tree is never disturbed.

## Common issues

- **`Permission denied (publickey)`** — wrong `IdentityFile`, or `.pem` permissions are too open (`chmod 400 ~/.ssh/sralnik-h100.pem`).
- **`Connection timed out`** — your laptop's public IP is not in the security group's allowlist. Aleksandr can add it:
  ```bash
  source ~/path/to/secrets.local.sh
  YOUR_IP=$(curl -s -4 ifconfig.me)
  aws ec2 authorize-security-group-ingress --region "$AWS_REGION" \
    --group-id "$SECURITY_GROUP_ID" --protocol tcp --port 22 --cidr "$YOUR_IP/32"
  ```
- **`Host key verification failed`** — the instance was relaunched and now has a different host key. Edit `~/.ssh/known_hosts` and remove the old entry for that IP, then reconnect.
- **Instance has been terminated (T+24h)** — the SSH endpoint is gone. Pull artefacts from S3 instead:
  ```bash
  source ~/path/to/secrets.local.sh
  aws s3 sync "$S3_RUNS"  ./runs-final/
  aws s3 sync "$S3_EVAL"  ./eval-final/
  ```
