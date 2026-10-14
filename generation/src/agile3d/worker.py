import argparse

from .pipeline import run_candidate
from .store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--retry-image", action="store_true")
    args = parser.parse_args()
    job = run_candidate(
        Store(args.root), args.job, args.token, retry_image=args.retry_image
    )
    raise SystemExit(1 if job["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
