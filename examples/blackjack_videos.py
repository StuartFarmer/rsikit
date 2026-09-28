"""Render Bitcoin or Blackjack EliteSearch leaders on training or held-out panels."""

import argparse
import asyncio
import html
import json
import math
import shutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from hashlib import sha256
from pathlib import Path

import gymnasium as gym
import numpy as np
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from PIL import Image

from rsikit import Executor
from rsikit.envs import BitcoinEnv, BlackjackEnv
from rsikit.envs.bitcoin import TRAIN_DATA, load_prices
from rsikit.envs.bitcoin_render import BitcoinRenderer
from rsikit.envs.blackjack_render import BlackjackRenderer


def evaluation_panel(
    experiment, *, split="training", seeds=None, data_path=None, start_date=None, end_date=None
):
    """Choose data independently of policy selection; never treat BTC seeds as OOS."""
    if split not in ("training", "holdout", "validation"):
        raise ValueError("Unknown evaluation split")
    heldout = split != "training"
    seeds = list(
        dict.fromkeys(
            seeds if seeds is not None else experiment["heldout_seeds" if heldout else "seeds"]
        )
    )
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Need nonnegative integer seeds")
    if experiment["env"] == "Bitcoin":
        data_path = Path(
            data_path
            or (
                Path(__file__).resolve().parents[1] / "data/bitcoin/validation.csv"
                if heldout
                else TRAIN_DATA
            )
        ).resolve()
        env = BitcoinEnv(data_path, start_date=start_date, end_date=end_date)
        if heldout and env._dates[0] < load_prices(TRAIN_DATA)[0][-1]:
            raise ValueError("Validation returns overlap the training period")
        panel = dict(
            split="validation" if heldout else "training",
            data_path=str(data_path),
            start_date=date.fromordinal(env._dates[0]).isoformat(),
            end_date=date.fromordinal(env._dates[-1]).isoformat(),
            data_sha256=sha256(json.dumps([env._dates, env._prices]).encode()).hexdigest(),
        )
    elif experiment["env"] == "Blackjack":
        if data_path or start_date or end_date:
            raise ValueError("Data paths and dates apply only to Bitcoin")
        if heldout and set(seeds) & set(experiment["seeds"]):
            raise ValueError("Training and held-out seeds must be disjoint")
        env = BlackjackEnv(shoes_per_episode=experiment.get("shoes_per_seed", 1))
        panel = dict(split="holdout" if heldout else "training")
    else:
        raise ValueError("Expected a Bitcoin or Blackjack run")
    return env, seeds, panel


def trace_environment(shoes=1, *, environment=None):
    # Local wrapper serializes by value; generated policy code stays in Docker.
    class Trace(gym.Wrapper):
        def reset(self, **kwargs):
            self.actions = []
            return self.env.reset(**kwargs)

        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            self.actions.append(action.tolist() if isinstance(action, np.ndarray) else int(action))
            if done or truncated:
                info["artifacts"] = {"actions.json": json.dumps(self.actions).encode()}
            return obs, reward, done, truncated, info

    return Trace(environment if environment is not None else BlackjackEnv(shoes_per_episode=shoes))


def render_policy(job):
    """Stream one frame per action; Bitcoin equity resets independently per seed."""
    path = Path(job["video"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    segments = []
    total_steps = 0
    total_reward = 0
    bitcoin = job.get("env") == "Bitcoin"
    renderer = (
        BitcoinRenderer(
            BitcoinEnv(
                job["data_path"], start_date=job.get("start_date"), end_date=job.get("end_date")
            ),
            policy_name=job["name"],
            split=job["split"],
        )
        if bitcoin
        else BlackjackRenderer(
            BlackjackEnv(shoes_per_episode=job["shoes"]), policy_name=job["name"]
        )
    )
    with renderer as env:
        with FFMPEG_VideoWriter(str(temporary), (1280, 720), 30, threads=1) as writer:
            for seed in job["seeds"]:
                history = None if bitcoin else env.points
                env.reset(seed=seed)
                if not bitcoin:
                    env.points = history
                trace = json.loads((Path(job["traces"]) / f"{seed}.json").read_text())
                reward = 0
                done = truncated = False
                first_frame = total_steps
                for action in trace["actions"]:
                    _, earned, done, truncated, _ = env.step(
                        np.array(action) if bitcoin else action
                    )
                    reward += earned
                    writer.write_frame(env.render())
                    total_steps += 1
                if not done or truncated or not math.isclose(reward, trace["reward"], abs_tol=1e-8):
                    raise ValueError("Replay does not match a complete episode trace")
                total_reward += reward
                segments.append(
                    {
                        "seed": seed,
                        "first_frame": first_frame,
                        "frames": len(trace["actions"]),
                        **(
                            dict(
                                final_equity=env.history[-1]["equity"],
                                total_fees=env.history[-1]["fees"],
                                max_drawdown=max(p["drawdown"] for p in env.history),
                                history=env.history,
                                trades=env.trades,
                            )
                            if bitcoin
                            else dict(rounds=env.unwrapped._rounds)
                        ),
                        "reward": reward,
                    }
                )
        if bitcoin:
            Image.fromarray(env.render()).save(path.with_suffix(".png"))
    with VideoFileClip(str(temporary)) as clip:
        assert clip.size == [1280, 720] and clip.fps == 30
        assert abs(clip.duration - total_steps / 30) < 0.02
        Image.fromarray(clip.get_frame(clip.duration / 2)).save(path.with_suffix(".jpg"))
    temporary.replace(path)
    return {
        "policy_id": job["policy_id"],
        "name": job["name"],
        "video": str(path),
        "frames": total_steps,
        "duration_seconds": total_steps / 30,
        "total_reward": total_reward,
        "segments": segments,
    }


def write_index(output, report):
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    sections = []
    for generation in sorted({r["generation"] for r in report["videos"]}):
        cards = []
        for row in report["videos"]:
            if row["generation"] != generation:
                continue
            video = row["video"]
            poster = str(Path(video).with_suffix(".jpg"))
            snapshot = str(Path(video).with_suffix(".png"))
            cards.append(
                f"<article><h3>#{row['rank']} · {html.escape(row['name'])}</h3>"
                f"<p>Search mean: <b>{row['search_mean']:+.4f}</b> · "
                f"Replay mean: <b>{row['total_reward'] / len(row['segments']):+g}</b></p>"
                f'<video controls preload="none" src="{video}" poster="{poster}"></video>'
                f"<p>{len(row['segments'])} seeds · {row['frames']:,} actions · "
                f"{row['duration_seconds']:.1f} seconds · "
                f'<a href="{video}" download>Download MP4</a>'
                + (
                    f' · <a href="{snapshot}">Final chart (last seed)</a>'
                    if report["env"] == "Bitcoin"
                    else ""
                )
                + "</p></article>"
            )
        sections.append(f"<h2>Generation {generation}</h2><section>{''.join(cards)}</section>")
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{report['env']} — {report['split']}</title><style>"
        "body{margin:32px auto;padding:0 24px;max-width:1440px;background:#fafaf8;"
        "color:#1f2933;font:16px system-ui;line-height:1.5}h3{overflow-wrap:anywhere}"
        "section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}"
        "article{background:white;padding:16px;border:1px solid #cad3d5;border-radius:8px}"
        "video{display:block;width:100%;aspect-ratio:16/9}a{color:#284b63}"
        "@media(max-width:800px){section{grid-template-columns:1fr}}</style>"
        f"<h1>{report['env']} leaders — {report['split']}</h1>"
        f"<p>Source: {html.escape(report['source_run'])}. "
        f"{len(report['seeds'])} seeds. 1280×720 · 30 fps · one action per frame. "
        "Policies are selected by training score; policy memory resets at each seed.</p>"
        + (
            f"<p>Market dates: {report['start_date']} to {report['end_date']}. "
            "Each seed starts with fresh cash. Seeds change policy randomness, not market data. "
            "Equity and buy-and-hold include trading fees and final liquidation. "
            "Daily equity/fills are in <a href='manifest.json'>manifest.json</a>.</p>"
            if report["env"] == "Bitcoin"
            else f"<p>{report['shoes_per_seed']} shoe(s) per seed; points accumulate across seeds.</p>"
        )
        + "".join(sections)
        + "</html>"
    )


async def export(
    source,
    output,
    *,
    top=4,
    workers=2,
    through_generation=None,
    generation=None,
    split="training",
    seeds=None,
    data_path=None,
    start_date=None,
    end_date=None,
):
    experiment = json.loads((source / "experiment.json").read_text())
    if experiment["max_steps"] is not None:
        raise ValueError("Expected a full-episode run")
    # Historic experiments predate multi-shoe episodes and must retain their rules.
    shoes = experiment.get("shoes_per_seed", 1)
    environment, seeds, panel = evaluation_panel(
        experiment,
        split=split,
        seeds=seeds,
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
    )
    context = dict(
        source_run=str(source.resolve()),
        env=experiment["env"],
        seeds=seeds,
        shoes_per_seed=shoes,
        **panel,
    )
    # A separate cache identity prevents training traces from masquerading as validation.
    cache_key = sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]
    verify_search = split == "training" and not (data_path or start_date or end_date)
    with sqlite3.connect((source / "run.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        organisms = {r["id"]: dict(r) for r in db.execute("SELECT * FROM elitesearch_organism")}
        selections = [
            {**organisms[oid], "generation": g["number"], "rank": rank}
            for g in db.execute(
                "SELECT * FROM elitesearch_generation WHERE status='completed' ORDER BY number"
            )
            if through_generation is None or g["number"] <= through_generation
            if generation is None or g["number"] == generation
            for rank, oid in enumerate(json.loads(g["elite_ids"])[:top], 1)
        ]
    if not selections:
        raise ValueError("No completed generation leaders")
    unique = {r["policy_id"]: r for r in selections}
    output.mkdir(parents=True, exist_ok=True)
    traces = output / "traces" / cache_key
    jobs = []
    for pid, row in unique.items():
        (traces / pid).mkdir(parents=True, exist_ok=True)
        jobs.extend(
            (pid, row["implementation"], seed)
            for seed in seeds
            if not (traces / pid / f"{seed}.json").exists()
        )
    print(
        f"Replaying {len(jobs)} episodes for {len(unique)} policies across {len(seeds)} seeds",
        flush=True,
    )
    with trace_environment(environment=environment) as env:
        async with Executor(
            concurrency=8,
            episode_timeout=experiment.get("episode_timeout", 60.0),
        ) as executor:
            completed = 0
            async for pid, seed, result in executor.evaluate(jobs, env):
                expected = json.loads(unique[pid]["seed_scores"]).get(str(seed))
                if (
                    verify_search
                    and expected is not None
                    and not math.isclose(result.total_reward, expected, abs_tol=1e-8)
                ):
                    raise ValueError(f"Training replay score differs: {pid}, seed {seed}")
                data = {
                    "actions": json.loads(result.artifacts["actions.json"]),
                    "reward": result.total_reward,
                }
                (traces / pid / f"{seed}.json").write_text(json.dumps(data))
                completed += 1
                if completed % 32 == 0:
                    print(f"Verified {completed}/{len(jobs)} seed scores", flush=True)
    render_jobs = []
    rendered = {}
    for pid, row in unique.items():
        cached = output / "policies" / cache_key / f"{pid}.json"
        if cached.exists():
            rendered[pid] = json.loads(cached.read_text())
        else:
            render_jobs.append(
                {
                    "policy_id": pid,
                    "name": row["name"],
                    "env": experiment["env"],
                    **panel,
                    "shoes": shoes,
                    "seeds": seeds,
                    "traces": str((traces / pid).resolve()),
                    "video": str((output / "policies" / cache_key / f"{pid}.mp4").resolve()),
                }
            )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(render_policy, job) for job in render_jobs]):
            result = future.result()
            pid = result["policy_id"]
            rendered[pid] = result
            Path(result["video"]).with_suffix(".json").write_text(json.dumps(result, indent=2))
            print(
                f"Rendered {len(rendered)}/{len(unique)}: {result['name']} ({result['frames']} actions)",
                flush=True,
            )
    report = {
        **context,
        "fps": 30,
        "resolution": [1280, 720],
        "initial_frame": False,
        "videos": [],
    }
    for row in selections:
        result = rendered[row["policy_id"]]
        if verify_search and seeds == experiment["seeds"]:
            if not math.isclose(result["total_reward"] / len(seeds), row["score"], abs_tol=1e-8):
                raise ValueError("Training replay mean differs from saved score")
        directory = output / f"generation-{row['generation']:02}"
        directory.mkdir(exist_ok=True)
        path = directory / f"rank-{row['rank']:02}-policy-{row['id']:03}.mp4"
        shutil.copy2(result["video"], path)
        shutil.copy2(Path(result["video"]).with_suffix(".jpg"), path.with_suffix(".jpg"))
        if experiment["env"] == "Bitcoin":
            shutil.copy2(Path(result["video"]).with_suffix(".png"), path.with_suffix(".png"))
        report["videos"].append(
            {
                **result,
                "generation": row["generation"],
                "rank": row["rank"],
                "organism_id": row["id"],
                "search_mean": row["score"],
                "video": str(path.relative_to(output)),
            }
        )
    write_index(output, report)
    print(f"Done: {output / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    generations = parser.add_mutually_exclusive_group()
    generations.add_argument("--through-generation", type=int)
    generations.add_argument("--generation", type=int, help="Export only this completed generation")
    parser.add_argument(
        "--split", choices=("training", "holdout", "validation"), default="training"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the saved seed panel")
    parser.add_argument(
        "--data-path", type=Path, help="Bitcoin CSV; validation must follow training"
    )
    parser.add_argument("--start-date", help="Inclusive Bitcoin start date, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive Bitcoin end date, YYYY-MM-DD")
    args = parser.parse_args()
    if min(args.top, args.workers) < 1:
        parser.error("top and workers must be positive")
    suffix = "all-seeds" if args.split == "training" else args.split
    output = args.output or args.run.with_name(args.run.name + f"-{suffix}-videos")
    asyncio.run(
        export(
            args.run,
            output,
            top=args.top,
            workers=args.workers,
            through_generation=args.through_generation,
            generation=args.generation,
            split=args.split,
            seeds=args.seeds,
            data_path=args.data_path,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    )


if __name__ == "__main__":
    main()
