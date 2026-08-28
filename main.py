import hashlib
import json
import math
import threading
from copy import deepcopy

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()

SAFE_MAX = 9007199254740991

FREEZE = "freeze"
SELECT = "select"

DAG = []

# File names are treated as paths, not arbitrary executable paths.
# The API itself does not impose a fixed extension list; it validates
# safe artifact filenames and exact UTF-8 content.
UNSAFE_FILENAME_PARTS = (
    "\x00",
)

UNSAFE_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".sh",
    ".bat",
    ".cmd",
    ".com",
    ".scr",
    ".msi",
    ".vbs",
    ".ps1",
}

FREEZE_CODES = {
    "INVALID_INPUT",
    "UNALLOWED_UNSUPPORTED_REASON",
    "NOT_LOADABLE",
    "CALIBRATION_MISMATCH",
    "TOKENIZER_MISMATCH",
}

SELECT_CODES = {
    "NOT_FROZEN",
    "INVALID_LINEAGE",
    "INVALID_POLICY",
    "INVALID_PREDICTIONS",
    "INVALID_MANIFEST",
    "AGGREGATE_FLOOR",
    "MISSING_SLICE",
    "SLICE_FLOOR",
    "SIZE_LIMIT",
    "LATENCY_LIMIT",
}

BINARY = {0, 1}

# ------------------------------------------------------------
# Persistent state.
# ------------------------------------------------------------

STORE = {}
LOCK = threading.RLock()


# ============================================================
# Deterministic JSON / hashing
# ============================================================

def u8(value):
    return value.encode("utf-8")


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_string(value):
    return sha256_bytes(u8(value))


def digest_array(values):
    return sha256_bytes(
        u8(canonical_json(values))
    )


def utf8_sort(values):
    return sorted(values, key=u8)


def unique_sorted_codes(values):
    return sorted(
        set(values),
        key=u8,
    )


def is_safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_MAX
    )


def is_positive_safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= SAFE_MAX
    )


def is_finite_number(value):
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(float(value))


def nonempty_string(value):
    return (
        isinstance(value, str)
        and len(value) > 0
    )


def valid_utf8_string(value):
    if not isinstance(value, str):
        return False

    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False

    return len(value) > 0


def fingerprint(value):
    return sha256_string(
        canonical_json(value)
    )


def http_error(status, code):
    return JSONResponse(
        status_code=status,
        content={"error": code},
    )


# ============================================================
# Filename validation
# ============================================================

def valid_filename(name):
    if not isinstance(name, str):
        return False

    if len(name) == 0:
        return False

    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False

    if any(
        part in name
        for part in UNSAFE_FILENAME_PARTS
    ):
        return False

    # No absolute paths.
    if name.startswith("/"):
        return False

    if name.startswith("\\"):
        return False

    # No path traversal.
    pieces = name.replace("\\", "/").split("/")

    if any(
        piece in ("", ".", "..")
        for piece in pieces
    ):
        return False

    # No Windows drive paths.
    if len(name) >= 2 and name[1] == ":":
        return False

    lowered = name.lower()

    for ext in UNSAFE_EXTENSIONS:
        if lowered.endswith(ext):
            return False

    return True


# ============================================================
# Freeze manifest
# ============================================================

def calculate_inventory(files):
    """
    Calculate inventory exclusively from the supplied exact
    UTF-8 strings.

    Output key order:
        name, bytes, sha256
    """

    inventory = []

    for filename in sorted(
        files.keys(),
        key=u8,
    ):
        content = files[filename]

        raw = u8(content)

        inventory.append({
            "name": filename,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        })

    return inventory


def calculate_package_digest(inventory):
    return sha256_bytes(
        u8(canonical_json(inventory))
    )


def validate_files_object(files):
    if not isinstance(files, dict):
        return False

    if len(files) == 0:
        return False

    seen = set()

    for filename, content in files.items():

        if not valid_filename(filename):
            return False

        if filename in seen:
            return False

        seen.add(filename)

        if not valid_utf8_string(content):
            return False

    return True


# ============================================================
# Freeze candidate
# ============================================================

def freeze_one_candidate(
    candidate,
    request_calibration,
    request_tokenizer,
    allowed_reasons,
):

    # Candidate itself must be an object.
    if not isinstance(candidate, dict):
        return {
            "name": "",
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ],
        }

    required = {
        "name",
        "files",
        "loadable",
        "calibrationDigest",
        "tokenizerDigest",
        "unsupportedReason",
    }

    if set(candidate.keys()) != required:

        name = candidate.get("name")

        return {
            "name": (
                name
                if isinstance(name, str)
                else ""
            ),
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ],
        }

    name = candidate["name"]

    if not valid_utf8_string(name):

        return {
            "name": (
                name
                if isinstance(name, str)
                else ""
            ),
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ],
        }

    files = candidate["files"]

    # --------------------------------------------------------
    # Invalid files => exactly empty manifest.
    # --------------------------------------------------------

    if not validate_files_object(files):

        return {
            "name": name,
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ],
        }

    if not isinstance(
        candidate["loadable"],
        bool,
    ):
        return {
            "name": name,
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ],
        }

    if not valid_utf8_string(
        candidate["calibrationDigest"]
    ):
        return {
            "name": name,
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ],
        }

    if not valid_utf8_string(
        candidate["tokenizerDigest"]
    ):
        return {
            "name": name,
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ],
        }

    unsupported = candidate[
        "unsupportedReason"
    ]

    if unsupported is not None:

        if not valid_utf8_string(
            unsupported
        ):
            return {
                "name": name,
                "status": "invalid",
                "inventory": [],
                "totalBytes": None,
                "packageDigest": None,
                "reasonCodes": [
                    "INVALID_INPUT"
                ],
            }

    # --------------------------------------------------------
    # Exact artifact inventory.
    # --------------------------------------------------------

    inventory = calculate_inventory(
        files
    )

    total_bytes = sum(
        entry["bytes"]
        for entry in inventory
    )

    package_digest = calculate_package_digest(
        inventory
    )

    codes = []

    # --------------------------------------------------------
    # Unsupported candidate.
    # --------------------------------------------------------

    if unsupported is not None:

        if unsupported not in allowed_reasons:
            codes.append(
                "UNALLOWED_UNSUPPORTED_REASON"
            )

    else:

        if candidate["loadable"] is not True:
            codes.append(
                "NOT_LOADABLE"
            )

        if (
            candidate["calibrationDigest"]
            != request_calibration
        ):
            codes.append(
                "CALIBRATION_MISMATCH"
            )

        if (
            candidate["tokenizerDigest"]
            != request_tokenizer
        ):
            codes.append(
                "TOKENIZER_MISMATCH"
            )

    codes = unique_sorted_codes(codes)

    if codes:

        return {
            "name": name,
            "status": "invalid",
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": codes,
        }

    if unsupported is not None:

        return {
            "name": name,
            "status": "unsupported",
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": [],
        }

    return {
        "name": name,
        "status": "frozen",
        "inventory": inventory,
        "totalBytes": total_bytes,
        "packageDigest": package_digest,
        "reasonCodes": [],
    }


# ============================================================
# Freeze validation
# ============================================================

def validate_freeze_request(body):

    required = {
        "phase",
        "freezeId",
        "calibrationDigest",
        "tokenizerDigest",
        "allowedUnsupportedReasons",
        "candidates",
    }

    if set(body.keys()) != required:
        return False

    if body["phase"] != FREEZE:
        return False

    if (
        not isinstance(
            body["freezeId"],
            str,
        )
        or len(body["freezeId"]) == 0
        or len(body["freezeId"]) > 128
    ):
        return False

    if not valid_utf8_string(
        body["calibrationDigest"]
    ):
        return False

    if not valid_utf8_string(
        body["tokenizerDigest"]
    ):
        return False

    allowed = body[
        "allowedUnsupportedReasons"
    ]

    if not isinstance(allowed, list):
        return False

    for reason in allowed:
        if not valid_utf8_string(reason):
            return False

    if len(allowed) != len(set(allowed)):
        return False

    candidates = body["candidates"]

    if not isinstance(candidates, list):
        return False

    if len(candidates) == 0:
        return False

    names = set()

    for candidate in candidates:

        if not isinstance(candidate, dict):
            return False

        name = candidate.get("name")

        if not valid_utf8_string(name):
            return False

        if name in names:
            return False

        names.add(name)

    return True


def create_freeze_response(body):

    if not validate_freeze_request(body):
        return None

    candidates = []

    for candidate in body["candidates"]:

        candidates.append(
            freeze_one_candidate(
                candidate,
                body["calibrationDigest"],
                body["tokenizerDigest"],
                set(
                    body[
                        "allowedUnsupportedReasons"
                    ]
                ),
            )
        )

    candidates.sort(
        key=lambda item: u8(
            item["name"]
        )
    )

    return {
        "freezeId": body["freezeId"],
        "candidates": candidates,
    }


# ============================================================
# Submitted frozen candidate validation
# ============================================================

def validate_submitted_candidate(
    candidate
):
    """
    Recompute every manifest property from the submitted
    inventory.

    There is deliberately no trust in:
      totalBytes
      packageDigest
    """

    if not isinstance(candidate, dict):
        return False, None, None

    required = {
        "name",
        "status",
        "inventory",
        "totalBytes",
        "packageDigest",
        "reasonCodes",
    }

    if set(candidate.keys()) != required:
        return False, None, None

    name = candidate["name"]

    if not valid_utf8_string(name):
        return False, None, None

    status = candidate["status"]

    if status not in (
        "frozen",
        "unsupported",
        "invalid",
    ):
        return False, None, None

    reason_codes = candidate[
        "reasonCodes"
    ]

    if not isinstance(
        reason_codes,
        list,
    ):
        return False, None, None

    if reason_codes != unique_sorted_codes(
        reason_codes
    ):
        return False, None, None

    # --------------------------------------------------------
    # Invalid frozen candidate has no artifact manifest.
    # --------------------------------------------------------

    if status == "invalid":

        if candidate["inventory"] != []:
            return False, None, None

        if candidate["totalBytes"] is not None:
            return False, None, None

        if candidate["packageDigest"] is not None:
            return False, None, None

        return True, None, None

    inventory = candidate[
        "inventory"
    ]

    if not isinstance(
        inventory,
        list,
    ):
        return False, None, None

    if len(inventory) == 0:
        return False, None, None

    normalized = []
    names = set()
    total = 0

    for item in inventory:

        if not isinstance(item, dict):
            return False, None, None

        if set(item.keys()) != {
            "name",
            "bytes",
            "sha256",
        }:
            return False, None, None

        name_i = item["name"]
        bytes_i = item["bytes"]
        hash_i = item["sha256"]

        if not valid_filename(name_i):
            return False, None, None

        if name_i in names:
            return False, None, None

        names.add(name_i)

        if not is_safe_int(bytes_i):
            return False, None, None

        if (
            not isinstance(hash_i, str)
            or len(hash_i) != 64
            or any(
                c not in "0123456789abcdef"
                for c in hash_i
            )
        ):
            return False, None, None

        normalized.append({
            "name": name_i,
            "bytes": bytes_i,
            "sha256": hash_i,
        })

        total += bytes_i

        if total > SAFE_MAX:
            return False, None, None

    # Exact UTF-8 ordering.
    expected_order = sorted(
        normalized,
        key=lambda item: u8(
            item["name"]
        ),
    )

    if normalized != expected_order:
        return False, None, None

    calculated_digest = (
        calculate_package_digest(
            normalized
        )
    )

    if candidate[
        "totalBytes"
    ] != total:
        return False, None, None

    if candidate[
        "packageDigest"
    ] != calculated_digest:
        return False, None, None

    return (
        True,
        total,
        calculated_digest,
    )


# ============================================================
# Selection policy
# ============================================================

def validate_policy(policy):

    required = {
        "maxBytes",
        "aggregateFloor",
        "requiredSlices",
        "maxLatencyMs",
        "candidateOrder",
    }

    if not isinstance(policy, dict):
        return False

    if set(policy.keys()) != required:
        return False

    if not is_safe_int(
        policy["maxBytes"]
    ):
        return False

    if not is_finite_number(
        policy["aggregateFloor"]
    ):
        return False

    if not (
        0 <= float(
            policy["aggregateFloor"]
        ) <= 1
    ):
        return False

    required_slices = policy[
        "requiredSlices"
    ]

    if not isinstance(
        required_slices,
        dict,
    ):
        return False

    for name, floor in required_slices.items():

        if not valid_utf8_string(name):
            return False

        if not is_finite_number(floor):
            return False

        if not (
            0 <= float(floor) <= 1
        ):
            return False

    if not is_finite_number(
        policy["maxLatencyMs"]
    ):
        return False

    if float(
        policy["maxLatencyMs"]
    ) < 0:
        return False

    order = policy[
        "candidateOrder"
    ]

    if not isinstance(order, list):
        return False

    if len(order) == 0:
        return False

    if any(
        not valid_utf8_string(name)
        for name in order
    ):
        return False

    if len(order) != len(set(order)):
        return False

    return True


# ============================================================
# Selection rows
# ============================================================

def validate_row(row):

    if not isinstance(row, dict):
        return False

    if set(row.keys()) != {
        "label",
        "slice",
        "predictions",
    }:
        return False

    label = row["label"]

    if (
        not isinstance(label, int)
        or isinstance(label, bool)
        or label not in BINARY
    ):
        return False

    if not valid_utf8_string(
        row["slice"]
    ):
        return False

    predictions = row[
        "predictions"
    ]

    if not isinstance(
        predictions,
        dict,
    ):
        return False

    return True


def prediction_for(
    row,
    candidate_name,
):
    predictions = row[
        "predictions"
    ]

    if candidate_name not in predictions:
        return None, False

    prediction = predictions[
        candidate_name
    ]

    if (
        not isinstance(
            prediction,
            int,
        )
        or isinstance(
            prediction,
            bool,
        )
        or prediction not in BINARY
    ):
        return None, False

    return prediction, True


# ============================================================
# Selection
# ============================================================

def perform_select(
    body,
    frozen_response,
):

    freeze_id = body[
        "freezeId"
    ]

    submitted = body[
        "candidates"
    ]

    policy = body[
        "policy"
    ]

    latencies = body[
        "latencies"
    ]

    rows = body[
        "rows"
    ]

    frozen = frozen_response[
        "candidates"
    ]

    # --------------------------------------------------------
    # Exact candidate array comparison.
    # --------------------------------------------------------

    lineage_match = (
        submitted == frozen
    )

    # --------------------------------------------------------
    # Frozen names.
    # --------------------------------------------------------

    frozen_names = [
        c["name"]
        for c in frozen
        if isinstance(c, dict)
        and isinstance(
            c.get("name"),
            str,
        )
    ]

    candidate_map = {
        c["name"]: c
        for c in frozen
        if isinstance(c, dict)
        and isinstance(
            c.get("name"),
            str,
        )
    }

    # --------------------------------------------------------
    # Policy.
    # --------------------------------------------------------

    policy_valid = validate_policy(
        policy
    )

    candidate_order = (
        policy.get("candidateOrder")
        if isinstance(policy, dict)
        else []
    )

    order_valid = (
        policy_valid
        and len(candidate_order)
        == len(frozen_names)
        and set(candidate_order)
        == set(frozen_names)
        and len(candidate_order)
        == len(set(candidate_order))
    )

    if order_valid:
        names = list(candidate_order)
    else:
        names = sorted(
            candidate_map.keys(),
            key=u8,
        )

    order_index = {
        name: i
        for i, name in enumerate(names)
    }

    # --------------------------------------------------------
    # Row validity.
    # --------------------------------------------------------

    rows_valid = (
        isinstance(rows, list)
        and len(rows) > 0
        and all(
            validate_row(row)
            for row in rows
        )
    )

    results = []

    for name in names:

        codes = []

        aggregate = None
        slice_values = {}
        total_bytes = None
        latency_ms = None

        candidate = candidate_map.get(
            name
        )

        # ----------------------------------------------------
        # Lineage.
        # ----------------------------------------------------

        if candidate is None:
            codes.append(
                "NOT_FROZEN"
            )

        if not lineage_match:
            codes.append(
                "INVALID_LINEAGE"
            )

        # ----------------------------------------------------
        # Policy.
        # ----------------------------------------------------

        if not policy_valid:
            codes.append(
                "INVALID_POLICY"
            )

        # ----------------------------------------------------
        # Artifact integrity.
        # ----------------------------------------------------

        manifest_valid = False

        if candidate is not None:

            (
                manifest_valid,
                calculated_bytes,
                _,
            ) = validate_submitted_candidate(
                candidate
            )

            if not manifest_valid:
                codes.append(
                    "INVALID_MANIFEST"
                )
            else:
                total_bytes = calculated_bytes

            if candidate.get(
                "status"
            ) != "frozen":
                codes.append(
                    "INVALID_LINEAGE"
                )

        # ----------------------------------------------------
        # Latency.
        # ----------------------------------------------------

        if (
            not isinstance(
                latencies,
                dict
            )
            or name not in latencies
        ):

            codes.append(
                "INVALID_LINEAGE"
            )

        else:

            latency = latencies[name]

            if (
                not is_finite_number(
                    latency
                )
                or float(latency) < 0
            ):
                codes.append(
                    "INVALID_LINEAGE"
                )
            else:
                latency_ms = float(
                    latency
                )

        # ----------------------------------------------------
        # Predictions.
        # ----------------------------------------------------

        predictions_valid = rows_valid

        if predictions_valid:

            for row in rows:

                _, ok = prediction_for(
                    row,
                    name,
                )

                if not ok:
                    predictions_valid = False
                    break

        if not predictions_valid:

            codes.append(
                "INVALID_PREDICTIONS"
            )

        else:

            correct = sum(
                1
                for row in rows
                if row[
                    "predictions"
                ][name]
                == row["label"]
            )

            aggregate = round(
                correct / len(rows),
                12,
            )

            # ------------------------------------------------
            # Slice accuracy.
            # ------------------------------------------------

            for slice_name, floor in (
                policy[
                    "requiredSlices"
                ].items()
                if policy_valid
                else []
            ):

                slice_rows = [
                    row
                    for row in rows
                    if row["slice"]
                    == slice_name
                ]

                if len(slice_rows) == 0:

                    codes.append(
                        "MISSING_SLICE:"
                        + slice_name
                    )

                    continue

                slice_correct = sum(
                    1
                    for row in slice_rows
                    if row[
                        "predictions"
                    ][name]
                    == row["label"]
                )

                value = round(
                    slice_correct
                    / len(slice_rows),
                    12,
                )

                slice_values[
                    slice_name
                ] = value

                if value < float(floor):
                    codes.append(
                        "SLICE_FLOOR:"
                        + slice_name
                    )

        # ----------------------------------------------------
        # Aggregate gate.
        # ----------------------------------------------------

        if (
            aggregate is not None
            and policy_valid
            and aggregate
            < float(
                policy[
                    "aggregateFloor"
                ]
            )
        ):
            codes.append(
                "AGGREGATE_FLOOR"
            )

        # ----------------------------------------------------
        # Size gate.
        # ----------------------------------------------------

        if (
            total_bytes is not None
            and policy_valid
            and total_bytes
            > policy["maxBytes"]
        ):
            codes.append(
                "SIZE_LIMIT"
            )

        # ----------------------------------------------------
        # Latency gate.
        # ----------------------------------------------------

        if (
            latency_ms is not None
            and policy_valid
            and latency_ms
            > float(
                policy["maxLatencyMs"]
            )
        ):
            codes.append(
                "LATENCY_LIMIT"
            )

        codes = unique_sorted_codes(
            codes
        )

        results.append({
            "name": name,
            "aggregate": aggregate,
            "slices": slice_values,
            "totalBytes": total_bytes,
            "latencyMs": latency_ms,
            "admitted": len(codes) == 0,
            "reasonCodes": codes,
        })

    # --------------------------------------------------------
    # Results follow candidateOrder.
    # UTF-8 name is fallback.
    # --------------------------------------------------------

    results.sort(
        key=lambda r: (
            order_index.get(
                r["name"],
                len(order_index),
            ),
            u8(r["name"]),
        )
    )

    # --------------------------------------------------------
    # Select admitted candidate.
    # --------------------------------------------------------

    admitted = [
        r
        for r in results
        if r["admitted"]
    ]

    if len(admitted) == 0:

        selected = None
        manifest = None

    else:

        winner = min(
            admitted,
            key=lambda r: (
                r["totalBytes"],
                r["latencyMs"],
                order_index.get(
                    r["name"],
                    len(order_index),
                ),
                u8(r["name"]),
            ),
        )

        selected = winner[
            "name"
        ]

        # Exactly the recorded frozen winner object.
        manifest = deepcopy(
            candidate_map[selected]
        )

    return {
        "freezeId": freeze_id,
        "selected": selected,
        "results": results,
        "packageManifest": manifest,
    }


# ============================================================
# POST /quantize
# ============================================================

@app.post("/quantize")
async def quantize(request: Request):

    try:
        body = await request.json()
    except Exception:
        return http_error(
            400,
            "INVALID_INPUT",
        )

    if not isinstance(body, dict):
        return http_error(
            400,
            "INVALID_INPUT",
        )

    phase = body.get("phase")

    # --------------------------------------------------------
    # Unknown/missing phase.
    # --------------------------------------------------------

    if phase not in (
        FREEZE,
        SELECT,
    ):
        return http_error(
            400,
            "INVALID_INPUT",
        )

    # ========================================================
    # FREEZE
    # ========================================================

    if phase == FREEZE:

        if not validate_freeze_request(
            body
        ):
            return http_error(
                400,
                "INVALID_INPUT",
            )

        freeze_id = body[
            "freezeId"
        ]

        req_fingerprint = fingerprint(
            body
        )

        with LOCK:

            existing = STORE.get(
                freeze_id
            )

            if existing is not None:

                if (
                    existing[
                        "fingerprint"
                    ]
                    != req_fingerprint
                ):
                    return http_error(
                        409,
                        "FREEZE_ID_CONFLICT",
                    )

                return existing[
                    "response"
                ]

            response = create_freeze_response(
                body
            )

            if response is None:
                return http_error(
                    400,
                    "INVALID_INPUT",
                )

            STORE[freeze_id] = {
                "fingerprint":
                    req_fingerprint,
                "response":
                    response,
            }

            return response

    # ========================================================
    # SELECT
    # ========================================================

    required = {
        "phase",
        "freezeId",
        "candidates",
        "policy",
        "latencies",
        "rows",
    }

    if set(body.keys()) != required:
        return http_error(
            400,
            "INVALID_INPUT",
        )

    if not isinstance(
        body["freezeId"],
        str,
    ):
        return http_error(
            400,
            "INVALID_INPUT",
        )

    # The contract explicitly requires arrays for both
    # candidates and rows and an object for policy.
    if (
        not isinstance(
            body["candidates"],
            list,
        )
        or len(body["candidates"]) == 0
        or not isinstance(
            body["rows"],
            list,
        )
        or len(body["rows"]) == 0
        or not isinstance(
            body["policy"],
            dict,
        )
        or not isinstance(
            body["latencies"],
            dict,
        )
    ):
        return http_error(
            400,
            "INVALID_INPUT",
        )

    freeze_id = body[
        "freezeId"
    ]

    with LOCK:
        frozen = STORE.get(
            freeze_id
        )

    # --------------------------------------------------------
    # Unknown freeze ID.
    # --------------------------------------------------------

    if frozen is None:

        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": [],
            "packageManifest": None,
        }

    return perform_select(
        body,
        frozen["response"],
    )


# ============================================================
# Simple health endpoint
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "endpoint": "/quantize",
    }
