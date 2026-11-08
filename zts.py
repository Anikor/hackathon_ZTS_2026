import json
import subprocess
import sys
import time
from pathlib import Path

HOST = (Path(__file__).resolve().parent / ".host").read_text().strip()
API = "https://hackathon.osace.ro/api"
ROOT = Path(__file__).resolve().parent
TOKEN = (ROOT / ".token").read_text().strip()
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30"]


def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def remote(cmd):
    return sh(SSH + [HOST, f'cd /d "%USERPROFILE%\\zts2026" && {cmd}'])


def status():
    out = subprocess.run(["curl", "-sS", f"{API}/team/status?token={TOKEN}"], capture_output=True, text=True).stdout
    for lvl, s in json.loads(out)["levels"].items():
        print(f"level {lvl}: {'OPEN' if s['is_open'] else 'closed'}  {s['submissions_used']}/{s['max_submissions']} used")


def run(level):
    t0 = time.time()
    url = f"{API}/datasets/{level}?token={TOKEN}&type="
    remote(f'curl -sSf -o data\\level{level}_train.csv "{url}train" && curl -sSf -o data\\level{level}_test.csv "{url}test"')
    remote(f"py -3.11 src\\solve.py --level {level} --data data --out submissions")
    models = "lgb,et,rf"
    fixed = ' --fixed "lgb+lgb+et"' if level == 3 else ""
    remote(f"py -3.11 src\\multi.py train --level {level} --data data --out submissions_l{level}x --models {models}")
    remote(f"py -3.11 src\\multi.py blend --level {level} --data data --out submissions_l{level}x{fixed}")
    if level == 3:
        remote("py -3.11 src\\solve.py --level 3 --data data --out submissions_l3x\\sub --probs submissions_l3x")
        remote("py -3.11 src\\solve.py --level 3 --data data --out submissions_l3x\\sub --probs submissions_l3x --l3-weights entropy=1")
    fetch(level)
    print(f"done in {time.time() - t0:.0f} s")


def fetch(level):
    dst = ROOT / f"remote_l{level}"
    dst.mkdir(exist_ok=True)
    files = [f"submissions/level{level}_submission.csv", f"submissions/level{level}_cv.json",
             f"submissions_l{level}x/level{level}_blend.csv", f"submissions_l{level}x/blend.json"]
    if level == 3:
        files += ["submissions_l3x/sub/level3_submission.csv", "submissions_l3x/sub/level3_cv.json",
                  "submissions_l3x/sub/level3_submission_entropy1.csv", "submissions_l3x/sub/level3_cv_entropy1.json"]
    for f in files:
        name = f.replace("/", "__")
        with open(dst / name, "wb") as fh:
            subprocess.run(SSH + [HOST, f'type "%USERPROFILE%\\zts2026\\{f.replace("/", chr(92))}"'], stdout=fh)
    print("fetched:", sorted(p.name for p in dst.iterdir()))


def submit(level, path):
    path = Path(path)
    out = subprocess.run(["curl", "-sS", "-F", f"file=@{path};type=text/csv", "-F", f"token={TOKEN}",
                          f"{API}/submit/{level}"], capture_output=True, text=True).stdout
    print(out)
    try:
        score = json.loads(out).get("score")
    except ValueError:
        score = None
    with open(ROOT / "submissions" / "attempts.md", "a") as fh:
        fh.write(f"| {time.strftime('%H:%M')} | {level} | {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
                 f"| zts.py submit | | {score} |\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "status":
        status()
    elif cmd == "run":
        run(int(sys.argv[2]))
    elif cmd == "fetch":
        fetch(int(sys.argv[2]))
    elif cmd == "submit":
        submit(int(sys.argv[2]), sys.argv[3])
