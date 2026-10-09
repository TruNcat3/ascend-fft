"""Parse and validate the `scopes:` timing line emitted by build/fft_check.

PR #2 stage 1 locks the field vocabulary and applicability matrix:

  h2d            logical-input H2D event span   (long chains only, else NA)
  device_chain   device compute-kernel span of the chain, transfers excluded
                 (host boundary: pass1+pass2 FFT spans; device boundary:
                  one contiguous 3xtranspose + 2xFFT + 1xtwiddle span)
  d2h            logical-output D2H event span  (long chains only, else NA)

Both boundary styles use the same names with the same semantics; ranges that
do not apply print the literal token NA.  The legacy `device_only` field is
rejected so a future reader cannot silently compare two different scopes.

Semantic invariant locked here: host_end_to_end (wall) >= h2d + device_chain +
d2h, per-iteration by construction (wall_i covers all three spans), with
slack only for host-scheduling/clock-domain noise.  If transfers were ever
folded back into device_chain, the component sum would swallow the whole wall
budget and the slack-capped inequality fails.
"""

import re

FIELDS = ("plan_setup", "first_use", "e2e_mean", "e2e_min",
          "h2d", "device_chain", "d2h", "reps")

_MODES = ("short", "long_host", "long_device")


def parse_scopes(line):
    """Parse one `scopes: ...` line into a dict.

    NA spans parse to None.  Raises ValueError on the retired `device_only`
    vocabulary, on missing fields, or on a malformed line.
    """
    if not isinstance(line, str) or not line.startswith("scopes:"):
        raise ValueError("not a scopes line: %r" % (line,))
    if "device_only" in line:
        raise ValueError(
            "retired device_only scope; expect h2d/device_chain/d2h fields")
    fields = {}
    m = re.search(r"plan_setup=([0-9.eE+-]+) us", line)
    if not m:
        raise ValueError("missing plan_setup: %r" % (line,))
    fields["plan_setup"] = float(m.group(1))
    m = re.search(r"first_use=([0-9.eE+-]+) us", line)
    if not m:
        raise ValueError("missing first_use: %r" % (line,))
    fields["first_use"] = float(m.group(1))
    m = re.search(r"host_end_to_end mean=([0-9.eE+-]+) min=([0-9.eE+-]+) us",
                  line)
    if not m:
        raise ValueError("missing host_end_to_end: %r" % (line,))
    fields["e2e_mean"] = float(m.group(1))
    fields["e2e_min"] = float(m.group(2))
    for name in ("h2d", "device_chain", "d2h"):
        m = re.search(r"\b%s=(NA|[0-9.eE+-]+(?: us)?)\b" % name, line)
        if not m:
            raise ValueError("missing %s: %r" % (name, line))
        raw = m.group(1)
        fields[name] = None if raw == "NA" else float(raw.split()[0])
    m = re.search(r"\breps=(\d+)\b", line)
    if not m:
        raise ValueError("missing reps: %r" % (line,))
    fields["reps"] = int(m.group(1))
    if set(fields) != set(FIELDS):
        raise ValueError("unexpected field set %r" % (sorted(fields),))
    return fields


def validate_scopes(fields, mode, slack_frac=0.10, slack_us=200.0):
    """Check names/applicability/ranges for `mode` in _MODES.

    Returns a list of human-readable problems; empty list == valid.
    """
    problems = []
    if mode not in _MODES:
        return ["unknown mode %r (expected one of %r)" % (mode, _MODES,)]
    for name in ("plan_setup", "first_use", "e2e_mean", "e2e_min"):
        v = fields.get(name)
        if v is None:
            problems.append("%s missing" % name)
        elif not (v > 0):
            problems.append("%s must be > 0, got %r" % (name, v))
    chain = fields.get("device_chain")
    if chain is None:
        problems.append("device_chain must be measured in every mode")
    elif not (chain > 0):
        problems.append("device_chain must be > 0, got %r" % (chain,))
    if fields.get("reps") is None or fields["reps"] < 1:
        problems.append("reps must be >= 1")
    if mode == "short":
        if fields.get("h2d") is not None:
            problems.append("short path must report h2d=NA, got %r"
                            % (fields.get("h2d"),))
        if fields.get("d2h") is not None:
            problems.append("short path must report d2h=NA, got %r"
                            % (fields.get("d2h"),))
    else:
        for name in ("h2d", "d2h"):
            v = fields.get(name)
            if v is None:
                problems.append("%s must be measured on %s chains" % (name, mode))
            elif not (v > 0):
                problems.append("%s must be > 0, got %r" % (name, v))
        measured = [v for v in (fields.get("h2d"), chain, fields.get("d2h"))
                    if v is not None]
        budget = sum(measured)
        for key in ("e2e_mean", "e2e_min"):
            wall = fields.get(key)
            if wall is None:
                continue
            slack = slack_frac * wall + slack_us
            if budget > wall + slack:
                problems.append(
                    "%s=%r below h2d+device_chain+d2h=%r (slack=%r): "
                    "transfers are not allowed inside device_chain"
                    % (key, wall, budget, slack))
    return problems


def infer_mode(fields, dev_boundary):
    """Convenience: mode from parsed fields + the AB_BOUNDARY setting."""
    if fields.get("h2d") is None:
        return "short"
    return "long_device" if dev_boundary else "long_host"
