#!/usr/bin/env python3
"""Fail closed when a measured hardware profile is used on another device."""
import argparse
import json
from pathlib import Path
import re
import sys


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def compatible(profile, device_name):
    expected = [profile.get("soc", ""), *profile.get("compatible_device_names", [])]
    actual = normalized(device_name)
    return bool(actual) and any(normalized(item) == actual for item in expected if item)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--target-soc", required=True,
                        help="SoC passed to the compiler (AB_SOC)")
    args = parser.parse_args(argv)
    try:
        profile = json.loads(args.profile.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read hardware profile: {error}")
    expected = [profile.get("soc", ""), *profile.get("compatible_device_names", [])]
    mismatches = []
    if not compatible(profile, args.device_name):
        mismatches.append(f"detected device {args.device_name!r}")
    if not compatible(profile, args.target_soc):
        mismatches.append(f"compiler target {args.target_soc!r}")
    if mismatches:
        print("hardware profile mismatch: " + " and ".join(mismatches)
              + "; expected one of "
              + ", ".join(repr(item) for item in expected if item), file=sys.stderr)
        return 1
    print(f"profile {profile.get('profile_id', args.profile.name)} matches device "
          f"{args.device_name} and compiler target {args.target_soc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
