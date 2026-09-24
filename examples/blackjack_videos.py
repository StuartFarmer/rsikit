"""Render EliteSearch generation leaders across all saved search seeds at 30 fps."""

import argparse
import asyncio
import html
import json
import math
import shutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from PIL import Image

from rsikit import Executor
from rsikit.envs import BlackjackEnv
from rsikit.envs.blackjack_render import BlackjackRenderer
from rsikit.sandbox.docker import DockerSandbox


def trace_environment(shoes):
    # Local wrapper serializes by value; generated policy code stays in Docker.
    class Trace(gym.Wrapper):
        def reset(self, **kwargs):
            self.actions = []
            return self.env.reset(**kwargs)

        def step(self, action):
            self.actions.append(int(action))
            obs, reward, done, truncated, info = self.env.step(action)
            if done or truncated:
                info["artifacts"] = {"actions.json": json.dumps(self.actions).encode()}
            return obs, reward, done, truncated, info

    return Trace(BlackjackEnv(shoes_per_episode=shoes))


def render_policy(job):
    """Stream exactly one frame per action, preserving the chart across seeds."""
    path = Path(job["video"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    segments = []
    total_steps = 0
    with BlackjackRenderer(
        BlackjackEnv(shoes_per_episode=job["shoes"]), policy_name=job["name"]
    ) as env:
        with FFMPEG_VideoWriter(str(temporary), (1280, 720), 30, threads=1) as writer:
            for seed in job["seeds"]:
                history = env.points
                env.reset(seed=seed)
                env.points = history
                trace = json.loads((Path(job["traces"]) / f"{seed}.json").read_text())
                reward = 0
                first_frame = total_steps
                for action in trace["actions"]:
                    _, earned, done, truncated, _ = env.step(action)
                    reward += earned
                    writer.write_frame(env.render())
                    total_steps += 1
                assert done and not truncated
                assert math.isclose(reward, trace["reward"])
                segments.append(
                    {
                        "seed": seed,
                        "first_frame": first_frame,
                        "frames": len(trace["actions"]),
                        "rounds": env.unwrapped._rounds,
                        "reward": reward,
                    }
                )
        total_reward = env.points[-1]
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
            cards.append(
                f"<article><h3>#{row['rank']} · {html.escape(row['name'])}</h3>"
                f"<p>Search mean: <b>{row['search_mean']:+.4f}</b> · "
                f"All-seed total: <b>{row['total_reward']:+g}</b></p>"
                f'<video controls preload="none" src="{video}" poster="{poster}"></video>'
                f"<p>{len(row['segments'])} seeds · {row['frames']:,} actions · "
                f"{row['duration_seconds']:.1f} seconds · "
                f'<a href="{video}" download>Download MP4</a></p></article>'
            )
        sections.append(f"<h2>Generation {generation}</h2><section>{''.join(cards)}</section>")
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Blackjack — all seeds</title><style>"
        "body{margin:32px auto;padding:0 24px;max-width:1440px;background:#fafaf8;"
        "color:#1f2933;font:16px system-ui;line-height:1.5}h3{overflow-wrap:anywhere}"
        "section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}"
        "article{background:white;padding:16px;border:1px solid #cad3d5;border-radius:8px}"
        "video{display:block;width:100%;aspect-ratio:16/9}a{color:#284b63}"
        "@media(max-width:800px){section{grid-template-columns:1fr}}</style>"
        "<h1>Blackjack leaders — all seeds</h1>"
        f"<p>Source: {html.escape(report['source_run'])}. "
        f"{len(report['seeds'])} seeds in order, {report['shoes_per_seed']} shoe(s) per seed. "
        "1280×720 · 30 fps · one action per frame. The chart accumulates points across "
        "all seeds; policy memory resets at each seed. Only completed generations are shown.</p>"
        + "".join(sections)
        + "</html>"
    )


async def export(source, output, *, top=4, workers=2, through_generation=None):
    experiment = json.loads((source / "experiment.json").read_text())
    if experiment["env"] != "Blackjack" or experiment["max_steps"] is not None:
        raise ValueError("Expected a full-episode Blackjack run")
    # Historic experiments predate multi-shoe episodes and must retain their rules.
    shoes = experiment.get("shoes_per_seed", 1)
    seeds = experiment["seeds"]
    with sqlite3.connect((source / "run.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        organisms = {r["id"]: dict(r) for r in db.execute("SELECT * FROM elitesearch_organism")}
        selections = [
            {**organisms[oid], "generation": g["number"], "rank": rank}
            for g in db.execute(
                "SELECT * FROM elitesearch_generation WHERE status='completed' ORDER BY number"
            )
            if through_generation is None or g["number"] <= through_generation
            for rank, oid in enumerate(json.loads(g["elite_ids"])[:top], 1)
        ]
    if not selections:
        raise ValueError("No completed generation leaders")
    unique = {r["policy_id"]: r for r in selections}
    output.mkdir(parents=True, exist_ok=True)
    traces = output / "traces"
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
    with trace_environment(shoes) as env:
        async with Executor(
            concurrency=8,
            sandbox=DockerSandbox(episode_timeout=experiment.get("episode_timeout", 60.0)),
        ) as executor:
            completed = 0
            async for pid, seed, result in executor.evaluate(jobs, env):
                expected = json.loads(unique[pid]["seed_scores"])[str(seed)]
                assert math.isclose(result.score, expected), (pid, seed, result.score, expected)
                data = {
                    "actions": json.loads(result.artifacts["actions.json"]),
                    "reward": result.score,
                }
                (traces / pid / f"{seed}.json").write_text(json.dumps(data))
                completed += 1
                if completed % 32 == 0:
                    print(f"Verified {completed}/{len(jobs)} seed scores", flush=True)
    render_jobs = []
    rendered = {}
    for pid, row in unique.items():
        cached = output / "policies" / f"{pid}.json"
        if cached.exists():
            rendered[pid] = json.loads(cached.read_text())
        else:
            render_jobs.append(
                {
                    "policy_id": pid,
                    "name": row["name"],
                    "shoes": shoes,
                    "seeds": seeds,
                    "traces": str((traces / pid).resolve()),
                    "video": str((output / "policies" / f"{pid}.mp4").resolve()),
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
        "source_run": str(source),
        "seeds": seeds,
        "shoes_per_seed": shoes,
        "fps": 30,
        "resolution": [1280, 720],
        "initial_frame": False,
        "videos": [],
    }
    for row in selections:
        result = rendered[row["policy_id"]]
        assert math.isclose(result["total_reward"] / len(seeds), row["score"])
        directory = output / f"generation-{row['generation']:02}"
        directory.mkdir(exist_ok=True)
        path = directory / f"rank-{row['rank']:02}-policy-{row['id']:03}.mp4"
        shutil.copy2(result["video"], path)
        shutil.copy2(Path(result["video"]).with_suffix(".jpg"), path.with_suffix(".jpg"))
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
    parser.add_argument("--through-generation", type=int)
    args = parser.parse_args()
    if min(args.top, args.workers) < 1:
        parser.error("top and workers must be positive")
    output = args.output or args.run.with_name(args.run.name + "-all-seeds-videos")
    asyncio.run(
        export(
            args.run,
            output,
            top=args.top,
            workers=args.workers,
            through_generation=args.through_generation,
        )
    )


if __name__ == "__main__":
    main()
