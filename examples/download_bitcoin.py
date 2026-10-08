"""Download ten calendar years of daily BTC/USD and reserve the last three years.

Run: python -m examples.download_bitcoin --as-of 2026-09-19
Only training prices are bundled with rsikit; validation stays outside the package.
"""

import argparse
import csv
import hashlib
import io
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from rsikit.envs.bitcoin import load_prices

ROOT = Path(__file__).resolve().parents[1]


def years_before(day, years):
    # February 29 maps to February 28 in non-leap years.
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def download(as_of, source=None):
    start, cutoff, end = years_before(as_of, 10), years_before(as_of, 3), as_of - timedelta(days=1)
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?" + urlencode(
        dict(
            assets="btc",
            metrics="PriceUSD",
            frequency="1d",
            start_time=start,
            end_time=end,
            page_size=10000,
            format="csv",
        )
    )
    if source:
        raw = Path(source).read_bytes()
    else:
        with urlopen(url, timeout=60) as response:
            raw = response.read()
    rows = [
        (row["time"][:10], row["PriceUSD"])
        for row in csv.DictReader(io.StringIO(raw.decode("utf-8")))
        if row["asset"] == "btc" and start.isoformat() <= row["time"][:10] <= end.isoformat()
    ]
    expected = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    if [day for day, _ in rows] != expected:
        raise ValueError(
            "Source does not cover every requested day; refusing an incomplete download"
        )
    paths = [ROOT / "rsikit/envs/data/bitcoin_train.csv", ROOT / "data/bitcoin/validation.csv"]
    panels = [
        [row for row in rows if row[0] <= cutoff.isoformat()],
        [row for row in rows if row[0] >= cutoff.isoformat()],
    ]
    outputs = []
    for path, panel in zip(paths, panels):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["date", "price_usd"])
                writer.writerows(panel)
            load_prices(temporary)
            outputs.append((temporary, path))
        except BaseException:
            temporary.unlink(missing_ok=True)
            for pending, _ in outputs:
                pending.unlink(missing_ok=True)
            raise
    for temporary, path in outputs:
        temporary.replace(path)
    metadata = {
        "source": url,
        "provider": "Coin Metrics Community API, PriceUSD, frequency=1d",
        "license": "https://creativecommons.org/licenses/by-nc/4.0/",
        "attribution": "https://coinmetrics.io/; https://github.com/coinmetrics/data",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of.isoformat(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "split": "Shared boundary price; no overlapping daily returns. Final day liquidates BTC.",
        "files": [
            dict(
                path=str(path.relative_to(ROOT)),
                rows=len(panel),
                start=panel[0][0],
                end=panel[-1][0],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path, panel in zip(paths, panels)
        ],
    }
    (ROOT / "data/bitcoin/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of", type=date.fromisoformat, default=datetime.now(timezone.utc).date()
    )
    parser.add_argument("--source", type=Path, help="Use a previously downloaded source CSV")
    args = parser.parse_args()
    print(json.dumps(download(args.as_of, args.source), indent=2))


if __name__ == "__main__":
    main()
