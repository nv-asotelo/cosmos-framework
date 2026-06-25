#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
from PIL import Image

try:
    import av
except Exception as exc:  # pragma: no cover - printed in report
    av = None
    AV_IMPORT_ERROR = str(exc)
else:
    AV_IMPORT_ERROR = ""


REASONING_FORMAT_INSTRUCTION = """Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""

ROBOT_ARM_TRAJECTORY_PROMPT = (
    'You are given the task "Move the tape into the basket". Specify the 2D trajectory your end effector should follow '
    'in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": '
    '"gripper trajectory"}.\n\nPrompt format:\nAnswer the question using the following format:\n<think>\nYour '
    "reasoning.\n</think>\nWrite your final answer immediately after the </think> tag."
)

SAMPLING_DEFAULTS = {
    "standard": {
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "temperature": 0.7,
    },
    "reasoning": {
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "temperature": 0.6,
    },
}


EXAMPLES = [
    {
        "id": "robotics-next-action",
        "title": "Robotics Next Action Prediction",
        "media_url": "/examples/agibot.mp4",
        "media_name": "agibot.mp4",
        "media_kind": "video",
        "user_prompt": "What can be the next immediate action?",
        "system_prompt": "You are a helpful assistant.",
        "reasoning": True,
        "parameters": {"frames_per_second": 4, "max_tokens": 4096},
    },
    {
        "id": "robot-arm",
        "title": "robot arm pick up stuff",
        "media_url": "/examples/robot_tape.png",
        "media_name": "robot_tape.png",
        "media_kind": "image",
        "user_prompt": ROBOT_ARM_TRAJECTORY_PROMPT,
        "system_prompt": "You are a helpful assistant.",
        "reasoning": True,
        "parameters": {
            "frames_per_second": 2,
            "max_tokens": 4096,
            "repetition_penalty": 1.2,
            "temperature": 0.3,
            "top_p": 0.3,
        },
    },
    {
        "id": "sdg-critic",
        "title": "SDG critic",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_rejection_sampling.mp4",
        "media_name": "sdg-critic.mp4",
        "media_kind": "video",
        "user_prompt": (
            "Approve or reject this generated video for inclusion in a dataset for physical world model ai training. "
            "It must perfectly adhere to physics, object permanence, and have no anomalies. Any issue or concern causes "
            "rejection. Answer with Approve or Reject only."
        ),
        "system_prompt": "You are a helpful assistant.",
        "reasoning": True,
        "parameters": {
            "frames_per_second": 4,
            "max_tokens": 4096,
            "repetition_penalty": 1.2,
            "temperature": 0.3,
            "top_p": 0.3,
        },
    },
    {
        "id": "warehouse",
        "title": "warehouse",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_warehouse.mp4",
        "media_name": "warehouse.mp4",
        "media_kind": "video",
        "user_prompt": "Which worker picked up the dropped box?",
        "system_prompt": "You are a helpful warehouse monitoring system.",
        "reasoning": True,
        "parameters": {
            "frames_per_second": 2,
            "max_tokens": 4096,
            "repetition_penalty": 1.2,
            "temperature": 0.3,
            "top_p": 0.3,
        },
    },
    {
        "id": "forklift",
        "title": "forklift load weight evaluation",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_forklift.jpg",
        "media_name": "forklift-load.jpg",
        "media_kind": "image",
        "user_prompt": (
            "Locate the bounding box of the load and determine if its size and weight of load within the forklift's "
            "limits. Estimate weights. Return all as json. Include json location, estimated weight of the load, and if "
            "it's in the limit."
        ),
        "system_prompt": "You are a helpful assistant.",
        "reasoning": False,
        "parameters": {
            "frames_per_second": 2,
            "max_tokens": 4096,
            "repetition_penalty": 1.2,
            "temperature": 0.3,
            "top_p": 0.3,
        },
    },
    {
        "id": "mail-package",
        "title": "mail package",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_mail_package.mp4",
        "media_name": "mail-package.mp4",
        "media_kind": "video",
        "user_prompt": "Is the person allowed to pick up the packages?",
        "system_prompt": "You are a helpful assistant.",
        "reasoning": True,
        "parameters": {
            "frames_per_second": 2,
            "max_tokens": 4096,
            "repetition_penalty": 1.2,
            "temperature": 0.3,
            "top_p": 0.8,
        },
    },
]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs(root):
    for name in ["media", "frames", "raw", "traces", "answers", "errors"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def prompt_for_reasoning(prompt, enabled):
    stripped = prompt.strip()
    if not stripped:
        return ""
    if enabled:
        if "<think>" in stripped or REASONING_FORMAT_INSTRUCTION in stripped:
            return prompt
        return f"{stripped}\n\n{REASONING_FORMAT_INSTRUCTION}"
    return prompt.replace(REASONING_FORMAT_INSTRUCTION, "").strip()


def merged_params(example):
    defaults = SAMPLING_DEFAULTS["reasoning" if example["reasoning"] else "standard"].copy()
    params = defaults
    params.update(example.get("parameters") or {})
    return params


def request_media(vite_origin, example, session):
    url = example["media_url"]
    name = example["media_name"]
    if url.startswith("/"):
        absolute = f"{vite_origin.rstrip('/')}{url}"
        response = session.get(absolute, timeout=120)
        response.raise_for_status()
        mime = response.headers.get("content-type", "").split(";")[0] or guess_mime(name)
        # This mirrors App.tsx: local videos are passed as an absolute stream URL, local images as data URLs.
        request_value = absolute if example["media_kind"] == "video" else bytes_to_data_url(response.content, mime)
        request_key = "video" if example["media_kind"] == "video" else "image"
        return {
            "bytes": response.content,
            "mime": mime,
            "request_key": request_key,
            "request_value": request_value,
            "request_source": "local-url" if example["media_kind"] == "video" else "local-data-url",
            "preview_url": absolute,
        }

    query = urlencode({"url": url, "name": name})
    response = session.get(f"{vite_origin.rstrip('/')}/api/example-media?{query}", timeout=300)
    response.raise_for_status()
    data = response.json()
    raw, mime = data_url_to_bytes(data["dataUrl"])
    request_key = "image" if mime.startswith("image/") else "video"
    return {
        "bytes": raw,
        "mime": mime,
        "request_key": request_key,
        "request_value": data["dataUrl"],
        "request_source": "vite-proxied-data-url",
        "preview_url": url,
    }


def guess_mime(name):
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def bytes_to_data_url(raw, mime):
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def data_url_to_bytes(value):
    match = re.match(r"^data:([^;,]+)[^,]*,(.*)$", value, re.S)
    if not match:
        raise ValueError("Expected data URL")
    return base64.b64decode(match.group(2)), match.group(1)


def ext_for_mime(mime, fallback):
    guessed = mimetypes.guess_extension(mime or "")
    if guessed:
        return guessed.replace(".jpe", ".jpg")
    return Path(fallback).suffix or ".bin"


def save_media_and_first_frame(root, example, media):
    media_ext = ext_for_mime(media["mime"], example["media_name"])
    media_path = root / "media" / f"{example['id']}{media_ext}"
    media_path.write_bytes(media["bytes"])
    frame_path = root / "frames" / f"{example['id']}-first-frame.jpg"

    if media["mime"].startswith("image/"):
        with Image.open(media_path) as image:
            image.convert("RGB").save(frame_path, "JPEG", quality=92)
        return str(media_path), str(frame_path), None

    if av is None:
        return str(media_path), None, f"PyAV unavailable: {AV_IMPORT_ERROR}"

    try:
        with av.open(str(media_path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return str(media_path), None, "No video stream found"
            for frame in container.decode(stream):
                frame.to_image().convert("RGB").save(frame_path, "JPEG", quality=92)
                return str(media_path), str(frame_path), None
        return str(media_path), None, "No decodable video frame found"
    except Exception as exc:
        return str(media_path), None, f"First-frame decode failed: {exc}"


def parse_sse_lines(lines):
    event = "message"
    data_lines = []
    for raw_line in lines:
        line = raw_line.decode("utf-8", "replace").rstrip("\n")
        if line.endswith("\r"):
            line = line[:-1]
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if data_lines:
        yield event, "\n".join(data_lines)


def call_vite_stream(vite_origin, payload, timeout):
    url = f"{vite_origin.rstrip('/')}/api/reason/stream"
    answer = []
    reasoning = []
    events = []
    raw = None
    usage = None
    states = []
    started = time.time()
    with requests.post(url, json=payload, stream=True, timeout=(60, timeout)) as response:
        response.raise_for_status()
        for event_name, data_text in parse_sse_lines(response.iter_lines(decode_unicode=False)):
            if data_text == "[DONE]":
                break
            try:
                data = json.loads(data_text)
            except Exception:
                data = {"_unparsed": data_text}
            events.append({"event": event_name, "data": data})
            if event_name == "delta":
                channel = data.get("channel")
                text = data.get("text") or ""
                if channel == "reasoning":
                    reasoning.append(text)
                elif channel == "answer":
                    answer.append(text)
            elif event_name == "raw":
                raw = data
            elif event_name == "usage":
                usage = data
            elif event_name == "state":
                states.append(data)
            elif event_name == "error":
                return {
                    "status": "error",
                    "error": data.get("message") or data,
                    "answer": "".join(answer),
                    "reasoning": "".join(reasoning),
                    "raw": raw,
                    "usage": usage,
                    "states": states,
                    "events": events,
                    "elapsed_seconds": time.time() - started,
                }
    return {
        "status": "success",
        "answer": "".join(answer),
        "reasoning": "".join(reasoning),
        "raw": raw,
        "usage": usage,
        "states": states,
        "events": events,
        "elapsed_seconds": time.time() - started,
    }


def strip_code_fence(text):
    value = (text or "").strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", value, re.S | re.I)
    return match.group(1).strip() if match else value


def json_validity(text):
    candidate = strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
        return True, parsed, None
    except Exception as exc:
        return False, None, str(exc)


def raw_shape_issues(raw, answer, reasoning):
    issues = []
    if not isinstance(raw, dict):
        return ["vite_missing_raw_event"]
    if raw.get("openai", {}).get("object") == "chat.completion.chunk" or raw.get("object") == "chat.completion.chunk":
        issues.append("vite_raw_shape_regression_chunk_object")
    openai = raw.get("openai") if isinstance(raw.get("openai"), dict) else raw
    if openai.get("object") != "chat.completion":
        issues.append(f"vite_raw_object_{openai.get('object')}")
    choices = openai.get("choices") or []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    raw_content = message.get("content", "")
    if "<think>" in raw_content.lower() or "</think>" in raw_content.lower():
        issues.append("vite_content_contains_think_tags")
    if reasoning and not (message.get("reasoning_content") or raw.get("reasoning")):
        issues.append("vite_reasoning_not_preserved_in_raw")
    if answer and answer.strip() and answer.strip() not in str(raw_content):
        issues.append("vite_raw_content_differs_from_answer_channel")
    return issues


def classify(example, result):
    issues = []
    answer = (result.get("answer") or "").strip()
    reasoning = (result.get("reasoning") or "").strip()
    raw = result.get("raw")
    issues.extend(raw_shape_issues(raw, answer, reasoning))
    if result.get("status") != "success":
        issues.append("request_error")
    if not answer and result.get("status") == "success":
        issues.append("empty_answer")
    if "<think>" in answer.lower() or "</think>" in answer.lower():
        issues.append("answer_contains_think_tags")
    if example["id"] == "forklift":
        ok, _, error = json_validity(answer)
        if not ok:
            issues.append(f"model_json_invalid:{error}")
    if example["id"] == "sdg-critic":
        normalized = re.sub(r"[^a-z]", "", answer.lower())
        if normalized not in {"approve", "reject"}:
            issues.append("model_not_approve_reject_only")
        if re.search(r"\b(gripper|blue block|yellow block|move my|reposition|coordinate|robot arm)\b", reasoning, re.I):
            issues.append("model_reasoning_topic_mismatch_robot_policy")
    if example["reasoning"] and not reasoning:
        issues.append("missing_reasoning_trace")
    return issues


def write_text(path, value):
    path.write_text(value or "", encoding="utf-8")


def run_example(root, vite_origin, model, example, session, timeout):
    started = utc_now()
    record = {
        "id": example["id"],
        "title": example["title"],
        "started_at": started,
        "media_url": example["media_url"],
        "media_kind": example["media_kind"],
        "reasoning_enabled": example["reasoning"],
        "issues": [],
    }
    try:
        media = request_media(vite_origin, example, session)
        media_path, frame_path, frame_error = save_media_and_first_frame(root, example, media)
        params = merged_params(example)
        payload = {
            "prompt": prompt_for_reasoning(example["user_prompt"], example["reasoning"]),
            "systemPrompt": example["system_prompt"],
            "model": model,
            media["request_key"]: media["request_value"],
            "params": params,
        }
        record.update(
            {
                "media_mime": media["mime"],
                "media_request_source": media["request_source"],
                "media_file": media_path,
                "first_frame": frame_path,
                "first_frame_error": frame_error,
                "params": params,
                "prompt": payload["prompt"],
                "system_prompt": payload["systemPrompt"],
            }
        )
        result = call_vite_stream(vite_origin, payload, timeout)
        record.update(result)
        record["issues"] = classify(example, result)
        record["completed_at"] = utc_now()
    except Exception as exc:
        err_path = root / "errors" / f"{example['id']}.txt"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        record.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback_file": str(err_path),
                "completed_at": utc_now(),
                "issues": ["runner_exception"],
            }
        )

    write_text(root / "traces" / f"{example['id']}-trace.txt", record.get("reasoning", ""))
    write_text(root / "answers" / f"{example['id']}-answer.txt", record.get("answer", ""))
    (root / "raw" / f"{example['id']}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def preview(text, limit=220):
    value = re.sub(r"\s+", " ", (text or "").strip())
    return value[: limit - 3] + "..." if len(value) > limit else value


def write_summary(root, records, active_model):
    summary = {
        "generated_at": utc_now(),
        "active_model": active_model,
        "example_count": len(records),
        "records": [
            {
                "id": r["id"],
                "title": r["title"],
                "status": r.get("status"),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "issues": r.get("issues", []),
                "first_frame": r.get("first_frame"),
                "trace_file": str(root / "traces" / f"{r['id']}-trace.txt"),
                "answer_file": str(root / "answers" / f"{r['id']}-answer.txt"),
                "raw_file": str(root / "raw" / f"{r['id']}.json"),
                "answer_preview": preview(r.get("answer")),
                "trace_preview": preview(r.get("reasoning")),
            }
            for r in records
        ],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# build.nvidia.com Example Smoke",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Active model: `{active_model}`",
        f"- Artifact root: `{root}`",
        "",
        "| Example | Status | Seconds | Issues | First frame |",
        "|---|---:|---:|---|---|",
    ]
    for r in records:
        seconds = r.get("elapsed_seconds")
        seconds_text = f"{seconds:.1f}" if isinstance(seconds, (int, float)) else ""
        issues = ", ".join(r.get("issues") or ["none"])
        first_frame = r.get("first_frame") or r.get("first_frame_error") or ""
        lines.append(f"| `{r['id']}` | {r.get('status', '')} | {seconds_text} | {issues} | `{first_frame}` |")
    lines.extend(["", "## Output Previews", ""])
    for r in records:
        lines.extend(
            [
                f"### {r['id']}",
                "",
                f"- Trace: `{root / 'traces' / (r['id'] + '-trace.txt')}`",
                f"- Answer: `{root / 'answers' / (r['id'] + '-answer.txt')}`",
                f"- Raw: `{root / 'raw' / (r['id'] + '.json')}`",
                f"- Answer preview: {preview(r.get('answer')) or '(empty)'}",
                f"- Trace preview: {preview(r.get('reasoning')) or '(empty)'}",
                "",
            ]
        )
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vite-origin", default="http://127.0.0.1:5173")
    parser.add_argument("--out", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--ids", default="", help="Comma-separated example IDs; defaults to all build.nvidia.com examples")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    out = Path(args.out or f"/var/local/home/horde/cosmos3-edge/smoke/build-nvidia-{int(time.time())}")
    ensure_dirs(out)
    session = requests.Session()

    active_model = args.model
    if not active_model:
        try:
            data = session.get(f"{args.vite_origin.rstrip('/')}/api/active-model", timeout=10).json()
            active_model = data.get("checkpoint") or data.get("display_name") or "nvidia/Cosmos3-Edge-Reasoner"
        except Exception:
            active_model = "nvidia/Cosmos3-Edge-Reasoner"

    wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
    examples = [item for item in EXAMPLES if not wanted or item["id"] in wanted]
    if wanted:
        found = {item["id"] for item in examples}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"Unknown example ids: {', '.join(missing)}")

    records = []
    progress_path = out / "progress.json"
    for index, example in enumerate(examples, 1):
        progress_path.write_text(
            json.dumps(
                {
                    "started_at": utc_now(),
                    "index": index,
                    "total": len(examples),
                    "current": example["id"],
                    "completed": [record["id"] for record in records],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{index}/{len(examples)}] {example['id']}...", flush=True)
        record = run_example(out, args.vite_origin, active_model, example, session, args.timeout)
        records.append(record)
        write_summary(out, records, active_model)
        print(f"[{index}/{len(examples)}] {example['id']} -> {record.get('status')} issues={record.get('issues')}", flush=True)

    progress_path.write_text(
        json.dumps(
            {
                "completed_at": utc_now(),
                "index": len(examples),
                "total": len(examples),
                "current": None,
                "completed": [record["id"] for record in records],
                "summary": str(out / "summary.md"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(out, records, active_model)
    print(str(out))


if __name__ == "__main__":
    main()
