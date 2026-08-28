import hashlib
import json
import math
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()

SAFE_MAX = 9007199254740991

DAG = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

PARENT = {
    "verify_data": None,
    "prepare": "verify_data",
    "train": "prepare",
    "evaluate": "train",
    "register": "evaluate",
    "publish": "register",
}

INPUT_NAMES = [
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
]

NODE_INPUTS = {
    "verify_data": [
        "generation",
        "checksum",
    ],
    "prepare": [
        "canonicalData",
        "prepareCode",
        "prepareConfig",
    ],
    "train": [
        "prepareArtifact",
        "trainCode",
        "trainConfig",
        "runtime",
    ],
    "evaluate": [
        "trainArtifact",
        "canonicalData",
        "evaluateCode",
        "evaluateConfig",
    ],
    "register": [
        "evaluateArtifact",
        "schemaDigest",
    ],
    "publish": [
        "registerArtifact",
        "publishConfig",
    ],
}

EVENT_STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

SUCCESS_STATUSES = {
    "succeeded",
}

FAILURE_STATUSES = {
    "retryable_failed",
    "terminal_failed",
}

ERROR_CODES = {
    "INVALID_REQUEST",
    "INVALID_EVENT",
    "EVENT_ID_CONFLICT",
    "REVISION_CONFLICT",
    "EVIDENCE_CONFLICT",
    "STATUS_CONFLICT",
}


# ============================================================
# Persistent process state
# ============================================================

STATE = {}
LOCK = threading.RLock()


# STATE[session] = {
#     "revision": int,
#     "inputs": dict,
#     "inputFingerprint": str,
#     "events": {
#         eventId: canonicalEventJson
#     },
#     "eventRecords": {
#         eventId: event
#     },
#     "nodes": {
#         node: {
#             "key": str,
#             "status": None | started | retryable_failed |
#                       terminal_failed | succeeded,
#             "attempt": int | None,
#             "artifactDigest": str | None,
#             "eventIds": [...],
#             "startEventId": str | None,
#         }
#     },
#     "cache": {
#         node: {
#             cacheKey: {
#                 "artifactDigest": str,
#                 "eventId": str,
#             }
#         }
#     }
# }


# ============================================================
# Helpers
# ============================================================

def utf8(value):
    return value.encode("utf-8")


def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def digest_json_array(values):
    return hashlib.sha256(
        utf8(compact(values))
    ).hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        utf8(compact(value))
    ).hexdigest()


def safe_positive_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= SAFE_MAX
    )


def nonempty_string(value):
    return (
        isinstance(value, str)
        and len(value) > 0
    )


def error(code):
    return JSONResponse(
        status_code=409,
        content={
            "error": code
        },
    )


def bad_request():
    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_REQUEST"
        },
    )


# ============================================================
# DAG / dependency helpers
# ============================================================

def node_ready(session_state, node):
    parent = PARENT[node]

    if parent is None:
        return True

    parent_state = session_state["nodes"][parent]

    return (
        parent_state["status"]
        == "succeeded"
    )


def parent_state(session_state, node):
    parent = PARENT[node]

    if parent is None:
        return None

    return session_state["nodes"][parent]


def dependency_values(session_state, node):
    """
    Construct the exact values used to calculate a node key.

    Parent artifacts are represented as:
      prepareArtifact
      trainArtifact
      evaluateArtifact
      registerArtifact
    """

    values = {}

    parent = PARENT[node]

    if parent is not None:
        parent_state = session_state["nodes"][parent]

        artifact_name = {
            "prepare": "prepareArtifact",
            "train": "trainArtifact",
            "evaluate": "evaluateArtifact",
            "register": "registerArtifact",
            "publish": "publishArtifact",
        }.get(node)

        if artifact_name is not None:
            values[artifact_name] = (
                parent_state["artifactDigest"]
            )

    # For the six node definitions, add the named inputs.
    for name in NODE_INPUTS[node]:

        if name in values:
            continue

        if name in session_state["inputs"]:
            values[name] = session_state[
                "inputs"
            ][name]

    return values


def calculate_node_key(
    session_state,
    node,
):
    values = dependency_values(
        session_state,
        node,
    )

    ordered_values = []

    for name in NODE_INPUTS[node]:
        ordered_values.append(
            values.get(name)
        )

    return digest_json_array(
        ordered_values
    )


def dependency_digests(
    session_state,
    node,
    cache_key,
):
    values = dependency_values(
        session_state,
        node,
    )

    result = {}

    for name in NODE_INPUTS[node]:
        result[name] = values.get(name)

    result["cacheKey"] = cache_key

    return result


# ============================================================
# State creation
# ============================================================

def make_node_state():
    return {
        "key": None,
        "status": None,
        "attempt": None,
        "artifactDigest": None,
        "eventIds": [],
        "startEventId": None,
    }


def make_state(
    revision,
    inputs,
    input_fingerprint,
):
    return {
        "revision": revision,
        "inputs": inputs,
        "inputFingerprint": input_fingerprint,
        "events": {},
        "eventRecords": {},
        "nodes": {
            node: make_node_state()
            for node in DAG
        },
        "cache": {
            node: {}
            for node in DAG
        },
    }


# ============================================================
# Request validation
# ============================================================

def validate_pipeline_request(body):

    if not isinstance(body, dict):
        return False

    required = {
        "session",
        "revision",
        "inputs",
        "events",
    }

    if not required.issubset(
        set(body.keys())
    ):
        return False

    if not nonempty_string(
        body["session"]
    ):
        return False

    if not safe_positive_int(
        body["revision"]
    ):
        return False

    if not isinstance(
        body["inputs"],
        dict,
    ):
        return False

    if not isinstance(
        body["events"],
        list,
    ):
        return False

    for name in INPUT_NAMES:

        if not nonempty_string(
            body["inputs"].get(name)
        ):
            return False

    return True


# ============================================================
# Event validation
# ============================================================

def validate_event(event):

    if not isinstance(event, dict):
        return False

    required = {
        "eventId",
        "revision",
        "node",
        "attempt",
        "status",
        "key",
        "artifactDigest",
        "receiptId",
    }

    if set(event.keys()) != required:
        return False

    if not nonempty_string(
        event["eventId"]
    ):
        return False

    if not safe_positive_int(
        event["revision"]
    ):
        return False

    if event["node"] not in DAG:
        return False

    if not safe_positive_int(
        event["attempt"]
    ):
        return False

    if event["status"] not in EVENT_STATUSES:
        return False

    if not nonempty_string(
        event["key"]
    ):
        return False

    status = event["status"]

    artifact = event["artifactDigest"]
    receipt = event["receiptId"]

    if status == "succeeded":

        if not nonempty_string(artifact):
            return False

    else:

        if artifact is not None:
            return False

    if status in (
        "succeeded",
    ) and event["node"] in (
        "register",
        "publish",
    ):

        expected = (
            "receipt:"
            + event["node"]
            + ":"
            + event["key"]
        )

        if receipt != expected:
            return False

    else:

        if receipt is not None:
            return False

    return True


# ============================================================
# Event transition logic
# ============================================================

def transition_event(
    session_state,
    event,
):
    node = event["node"]
    node_state = session_state[
        "nodes"
    ][node]

    incoming_status = event[
        "status"
    ]
    incoming_attempt = event[
        "attempt"
    ]
    incoming_key = event["key"]

    current_key = node_state[
        "key"
    ]
    current_status = node_state[
        "status"
    ]
    current_attempt = node_state[
        "attempt"
    ]

    # --------------------------------------------------------
    # Event for a stale/different key is ignored unless it
    # attempts to contradict a current immutable success.
    # --------------------------------------------------------

    if current_key is not None:
        if incoming_key != current_key:

            # A successful cached/current state is immutable.
            if current_status == "succeeded":
                return "ignore", None

            # A node may receive an event only for its current key.
            return "ignore", None

    # --------------------------------------------------------
    # No current state.
    # --------------------------------------------------------

    if current_status is None:

        if incoming_status == "started":
            if incoming_attempt != 1:
                return "ignore", None

            node_state["key"] = incoming_key
            node_state["status"] = "started"
            node_state["attempt"] = 1
            node_state["artifactDigest"] = None
            node_state["eventIds"].append(
                event["eventId"]
            )
            node_state["startEventId"] = (
                event["eventId"]
            )

            return "accept", None

        # Completion without start is ignored.
        return "ignore", None

    # --------------------------------------------------------
    # Existing started state.
    # --------------------------------------------------------

    if current_status == "started":

        if (
            incoming_attempt
            == current_attempt
            and incoming_status in (
                "succeeded",
                "retryable_failed",
                "terminal_failed",
            )
        ):

            node_state["status"] = (
                incoming_status
            )

            node_state["artifactDigest"] = (
                event["artifactDigest"]
            )

            node_state["eventIds"].append(
                event["eventId"]
            )

            return "accept", None

        # Lower attempt is ignored.
        if incoming_attempt < current_attempt:
            return "ignore", None

        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # Retryable failure.
    # --------------------------------------------------------

    if current_status == "retryable_failed":

        if (
            incoming_status == "started"
            and incoming_attempt
            == current_attempt + 1
        ):

            node_state["status"] = "started"
            node_state["attempt"] = (
                incoming_attempt
            )
            node_state["artifactDigest"] = None
            node_state["eventIds"].append(
                event["eventId"]
            )
            node_state["startEventId"] = (
                event["eventId"]
            )

            return "accept", None

        if incoming_attempt < current_attempt:
            return "ignore", None

        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # Terminal failure.
    # --------------------------------------------------------

    if current_status == "terminal_failed":
        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # Successful state.
    # --------------------------------------------------------

    if current_status == "succeeded":

        if (
            incoming_status == "succeeded"
            and incoming_key == current_key
        ):

            if (
                event["artifactDigest"]
                != node_state["artifactDigest"]
            ):
                return (
                    "conflict",
                    "EVIDENCE_CONFLICT",
                )

            return "conflict", "STATUS_CONFLICT"

        return "conflict", "STATUS_CONFLICT"

    return "conflict", "STATUS_CONFLICT"


# ============================================================
# Cache handling
# ============================================================

def cache_success(
    session_state,
    node,
    key,
    artifact,
    event_id,
):
    cache = session_state[
        "cache"
    ][node]

    existing = cache.get(key)

    if existing is not None:

        if (
            existing["artifactDigest"]
            != artifact
        ):
            return False

        return True

    cache[key] = {
        "artifactDigest": artifact,
        "eventId": event_id,
    }

    return True


def apply_cached_state(
    session_state,
    node,
    key,
):
    cached = session_state[
        "cache"
    ][node].get(key)

    if cached is None:
        return False

    node_state = session_state[
        "nodes"
    ][node]

    node_state["key"] = key
    node_state["status"] = "succeeded"
    node_state["attempt"] = None
    node_state["artifactDigest"] = (
        cached["artifactDigest"]
    )
    node_state["eventIds"] = [
        cached["eventId"]
    ]
    node_state["startEventId"] = None

    return True


# ============================================================
# Process one event
# ============================================================

def process_event(
    session_state,
    event,
    accepted,
    ignored,
):
    event_id = event[
        "eventId"
    ]

    canonical_event = compact(
        event
    )

    # --------------------------------------------------------
    # Exact replay.
    # --------------------------------------------------------

    existing = session_state[
        "events"
    ].get(event_id)

    if existing is not None:

        if existing == canonical_event:
            ignored.append(event_id)
            return None

        return "EVENT_ID_CONFLICT"

    # --------------------------------------------------------
    # Revision mismatch: ignored and does not consume ID.
    # --------------------------------------------------------

    if event["revision"] != session_state[
        "revision"
    ]:
        ignored.append(event_id)
        return None

    # --------------------------------------------------------
    # Node must be ready.
    # --------------------------------------------------------

    node = event["node"]

    if not node_ready(
        session_state,
        node,
    ):
        ignored.append(event_id)
        return None

    # --------------------------------------------------------
    # Calculate current key.
    # --------------------------------------------------------

    key = calculate_node_key(
        session_state,
        node,
    )

    # Wrong key is ignored and does not consume ID.
    if event["key"] != key:
        ignored.append(event_id)
        return None

    # --------------------------------------------------------
    # Immutable cache conflict.
    # --------------------------------------------------------

    cached = session_state[
        "cache"
    ][node].get(key)

    if cached is not None:

        if event["status"] == "succeeded":

            if (
                event["artifactDigest"]
                != cached["artifactDigest"]
            ):
                return "EVIDENCE_CONFLICT"

            # Same immutable evidence but a new event
            # is still not a valid state transition.
            return "STATUS_CONFLICT"

        return "STATUS_CONFLICT"

    # --------------------------------------------------------
    # Apply transition.
    # --------------------------------------------------------

    result, conflict = transition_event(
        session_state,
        event,
    )

    if result == "ignore":
        ignored.append(event_id)
        return None

    if result == "conflict":
        return conflict

    # --------------------------------------------------------
    # Record event only after successful transition.
    # --------------------------------------------------------

    session_state[
        "events"
    ][event_id] = canonical_event

    session_state[
        "eventRecords"
    ][event_id] = dict(event)

    accepted.append(event_id)

    # --------------------------------------------------------
    # Successful artifact becomes immutable cache evidence.
    # --------------------------------------------------------

    if event["status"] == "succeeded":

        ok = cache_success(
            session_state,
            node,
            key,
            event["artifactDigest"],
            event_id,
        )

        if not ok:
            return "EVIDENCE_CONFLICT"

    return None


# ============================================================
# Response construction
# ============================================================

def node_output(
    session_state,
    node,
):
    node_state = session_state[
        "nodes"
    ][node]

    key = calculate_node_key(
        session_state,
        node,
    )

    # --------------------------------------------------------
    # Cache hit.
    # --------------------------------------------------------

    cached = session_state[
        "cache"
    ][node].get(key)

    if cached is not None:

        return {
            "node": node,
            "action": "reuse",
            "reasonCodes": [
                "CACHE_HIT"
            ],
            "dependencyDigests":
                dependency_digests(
                    session_state,
                    node,
                    key,
                ),
            "triggeringEventIds": [
                cached["eventId"]
            ],
        }

    # --------------------------------------------------------
    # Current node state.
    # --------------------------------------------------------

    if (
        node_state["key"] == key
        and node_state["status"] == "started"
    ):

        return {
            "node": node,
            "action": "block",
            "reasonCodes": [
                "RUNNING"
            ],
            "dependencyDigests":
                dependency_digests(
                    session_state,
                    node,
                    key,
                ),
            "triggeringEventIds": [
                node_state[
                    "startEventId"
                ]
            ]
            if node_state[
                "startEventId"
            ]
            else [],
        }

    if (
        node_state["key"] == key
        and node_state["status"]
        == "terminal_failed"
    ):

        return {
            "node": node,
            "action": "block",
            "reasonCodes": [
                "TERMINAL_FAILURE"
            ],
            "dependencyDigests":
                dependency_digests(
                    session_state,
                    node,
                    key,
                ),
            "triggeringEventIds":
                list(
                    node_state[
                        "eventIds"
                    ]
                ),
        }

    if (
        node_state["key"] == key
        and node_state["status"]
        == "retryable_failed"
    ):

        return {
            "node": node,
            "action": "rerun",
            "reasonCodes": [
                "RETRYABLE_FAILURE"
            ],
            "dependencyDigests":
                dependency_digests(
                    session_state,
                    node,
                    key,
                ),
            "triggeringEventIds":
                list(
                    node_state[
                        "eventIds"
                    ]
                ),
        }

    # --------------------------------------------------------
    # Upstream status.
    # --------------------------------------------------------

    parent = PARENT[node]

    if parent is not None:

        parent_state = session_state[
            "nodes"
        ][parent]

        parent_key = calculate_node_key(
            session_state,
            parent,
        )

        parent_cache = session_state[
            "cache"
        ][parent].get(parent_key)

        if parent_cache is None:

            if (
                parent_state["status"]
                == "terminal_failed"
            ):

                return {
                    "node": node,
                    "action": "block",
                    "reasonCodes": [
                        "UPSTREAM_TERMINAL"
                    ],
                    "dependencyDigests":
                        dependency_digests(
                            session_state,
                            node,
                            key,
                        ),
                    "triggeringEventIds":
                        list(
                            parent_state[
                                "eventIds"
                            ]
                        ),
                }

            return {
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_PENDING"
                ],
                "dependencyDigests":
                    dependency_digests(
                        session_state,
                        node,
                        key,
                    ),
                "triggeringEventIds":
                    list(
                        parent_state[
                            "eventIds"
                        ]
                    ),
            }

    # --------------------------------------------------------
    # Ready but no cache.
    # --------------------------------------------------------

    return {
        "node": node,
        "action": "rerun",
        "reasonCodes": [
            "CACHE_MISS"
        ],
        "dependencyDigests":
            dependency_digests(
                session_state,
                node,
                key,
            ),
        "triggeringEventIds":
            list(
                node_state[
                    "eventIds"
                ]
            ),
    }


def build_response(
    session_state,
    accepted,
    ignored,
):
    return {
        "revision": session_state[
            "revision"
        ],
        "acceptedEventIds": accepted,
        "ignoredEventIds": ignored,
        "nodes": [
            node_output(
                session_state,
                node,
            )
            for node in DAG
        ],
    }


# ============================================================
# Main endpoint
# ============================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await request.json()
    except Exception:
        return bad_request()

    if not validate_pipeline_request(
        body
    ):
        return bad_request()

    session = body[
        "session"
    ]

    revision = body[
        "revision"
    ]

    inputs = body[
        "inputs"
    ]

    events = body[
        "events"
    ]

    # --------------------------------------------------------
    # Validate every event before mutation.
    #
    # This guarantees that malformed events cannot partially
    # mutate the state.
    # --------------------------------------------------------

    for event in events:

        if not validate_event(event):
            return JSONResponse(
                status_code=409,
                content={
                    "error":
                        "INVALID_EVENT"
                },
            )

    # --------------------------------------------------------
    # Canonical request fingerprint.
    #
    # Extra metadata is deliberately included.
    # --------------------------------------------------------

    input_fingerprint = fingerprint(
        inputs
    )

    with LOCK:

        existing = STATE.get(
            session
        )

        # ====================================================
        # First request for this session
        # ====================================================

        if existing is None:

            session_state = make_state(
                revision,
                dict(inputs),
                input_fingerprint,
            )

            accepted = []
            ignored = []

            # Process events sequentially.
            for event in events:

                conflict = process_event(
                    session_state,
                    event,
                    accepted,
                    ignored,
                )

                if conflict is not None:
                    return error(
                        conflict
                    )

            STATE[session] = (
                session_state
            )

            return build_response(
                session_state,
                accepted,
                ignored,
            )

        # ====================================================
        # Existing session
        # ====================================================

        # Same revision requires exactly identical inputs,
        # including extra metadata.
        if revision == existing[
            "revision"
        ]:

            if (
                input_fingerprint
                != existing[
                    "inputFingerprint"
                ]
            ):
                return error(
                    "REVISION_CONFLICT"
                )

            # Work on a deep copy so a batch conflict can
            # atomically roll back everything.
            session_state = json.loads(
                json.dumps(
                    existing,
                    ensure_ascii=False,
                )
            )

            accepted = []
            ignored = []

            for event in events:

                conflict = process_event(
                    session_state,
                    event,
                    accepted,
                    ignored,
                )

                if conflict is not None:
                    return error(
                        conflict
                    )

            STATE[session] = (
                session_state
            )

            return build_response(
                session_state,
                accepted,
                ignored,
            )

        # ====================================================
        # New revision
        # ====================================================

        if revision > existing[
            "revision"
        ]:

            # Successful content-addressed cache entries survive
            # revision changes.
            preserved_cache = (
                existing["cache"]
            )

            session_state = make_state(
                revision,
                dict(inputs),
                input_fingerprint,
            )

            session_state[
                "cache"
            ] = json.loads(
                json.dumps(
                    preserved_cache,
                    ensure_ascii=False,
                )
            )

            accepted = []
            ignored = []

            for event in events:

                conflict = process_event(
                    session_state,
                    event,
                    accepted,
                    ignored,
                )

                if conflict is not None:
                    return error(
                        conflict
                    )

            STATE[session] = (
                session_state
            )

            return build_response(
                session_state,
                accepted,
                ignored,
            )

        # ----------------------------------------------------
        # Older revision.
        # ----------------------------------------------------

        # The request itself is an older revision. Its events
        # are ignored, but it must not mutate current state.
        ignored = [
            event["eventId"]
            for event in events
        ]

        return build_response(
            existing,
            [],
            ignored,
        )


# ============================================================
# Health endpoint
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "endpoint": "/pipeline",
    }
