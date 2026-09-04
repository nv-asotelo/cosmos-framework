# Cosmos3 Edge Reasoner staging report

This staging note captures the dual-host install and Spark smoke results for
`nvidia/Cosmos3-Edge-Reasoner` using the reasoner Vite surface from
`nv-asotelo/cosmos-cookbook:claude-skills-drop`.

The target PR is intentionally evidence-first: it preserves the examples,
first-frame screenshots, exact model traces, final answers, raw normalized Vite
responses, and a smoke runner so the bugs can be reproduced before deciding
which patches belong in `cosmos-framework`, `vllm-cosmos3`, or the cookbook UI.

## Environment

| Item | Value |
|---|---|
| Primary host | Spark, `horde@10.57.201.135` |
| Install root | `/var/local/home/horde/cosmos3-edge` |
| Checkpoint | `nvidia/Cosmos3-Edge-Reasoner` |
| Checkpoint commit | `9b4c028a5eb7d500d10a7f2fd7f0e7bc9c1abb77` |
| Vite endpoint | `http://10.57.201.135:5173` |
| Backend endpoint | `http://127.0.0.1:8000/v1` on the Spark host |
| Backend implementation used for smoke | Transformers/FastAPI shim |
| Evidence root in this repo | `docs/assets/cosmos3-edge-reasoner-smoke/` |
| Smoke runner | `tools/cosmos3_edge_build_smoke.py` |

## Summary

Spark can run the reasoner through the Vite app. The six visible
`build.nvidia.com` examples all completed successfully in the smoke sweep.

The current Vite patches fix the earlier UI/raw-output symptoms: every smoke
result has final raw object `chat.completion`, not `chat.completion.chunk`;
final `message.content` has no `<think>` tags; and reasoning is preserved in a
separate reasoning field. The remaining prominent issue is model-side: the
`sdg-critic` final answer is `Reject`, but the reasoning trace is an unrelated
robot gripper policy trace.

![First-frame contact sheet](assets/cosmos3-edge-reasoner-smoke/first-frame-contact-sheet.jpg)

## Smoke results

| Example | Media | Seconds | Result | Issue |
|---|---:|---:|---|---|
| `robotics-next-action` | video | 12.5 | success | none |
| `robot-arm` | image | 12.9 | success | returns JSON in a Markdown code fence |
| `sdg-critic` | video | 14.7 | success | reasoning trace is unrelated robot policy |
| `warehouse` | video | 37.6 | success | none |
| `forklift` | image | 114.0 | success | slow, but valid JSON |
| `mail-package` | video | 87.9 | success | over-infers permission from weak visual evidence |

The exact response objects are in
`docs/assets/cosmos3-edge-reasoner-smoke/raw/`. The first-frame screenshots are
in `docs/assets/cosmos3-edge-reasoner-smoke/frames/`.

## Bugs found and patch plan

| Bug | Evidence | Patch |
|---|---|---|
| Example videos did not load on Spark. | The browser examples initially showed missing/failing media. The smoke runner now verifies real assets and writes first frames under `frames/`. | Run `git lfs pull` for `apps/nvidia-build-reason-vite/public/examples` during deploy, or copy real assets into both `public/examples` and `dist/examples`. Add a startup check that rejects Git LFS pointer files by size/content before Vite starts. |
| Browser showed generic `fetch failed`. | The page only surfaced a generic error, hiding whether the failure was media fetch, Vite proxy, or backend stream. | Return structured SSE `error` events with the upstream phase and redacted payload. In the UI, render the structured error instead of collapsing it to `fetch failed`. |
| Vite JSON tab displayed `chat.completion.chunk` instead of a final JSON response. | Raw smoke outputs now show final `chat.completion` for all six examples. Earlier browser evidence showed chunk objects. | In the Vite stream route, accumulate streamed deltas, then emit a normalized final object: `object: "chat.completion"`, `choices[0].message.content = answer`, and `choices[0].message.reasoning_content = reasoning`. Apply the same normalization in the shared non-stream client. |
| Reasoning trace and final response were rendered together. | Smoke raw files show `message.content` without `<think>` tags and reasoning preserved separately. | Split `delta.reasoning_content`/explicit reasoning from `delta.content`; do not display combined `<think>...</think>` content as the answer. Keep reasoning in a separate UI panel or JSON field. |
| Backend imposed its own response cap after the user asked not to limit it. | Spark backend was patched to default to no extra backend clamp. The smoke payloads still include example `max_tokens=4096` because the Vite opt-in forwarding path was enabled. | Use `MAX_NEW_TOKENS_LIMIT=0` as the backend default, and clamp only when the env var is greater than zero. For fully uncapped app runs, unset `REASONER_SEND_MAX_TOKENS` or leave it disabled so Vite does not forward example `max_tokens`. |
| L40S x86 host is fragile for this checkpoint. | The L40S path required a CUDA 12.4/driver 550-compatible environment and could not build the mamba kernels cleanly. | Treat L40S as best-effort unless its driver/CUDA stack is upgraded. Keep Spark as the CUDA 13 path. If L40S support is required, add a documented Transformers fallback and a clear mamba-kernel compatibility gate. |
| `sdg-critic` reasoning is unrelated to the prompt. | Exact trace below begins with gripper movement, while the prompt asks for dataset accept/reject. | This is model/prompt behavior, not a Vite formatting issue. Short-term UI patch: set this example to non-reasoning mode or hide the trace by default. Model/prompt patch: evaluate a stricter critic prompt and schema, then checkpoint-side follow-up if the trace remains robotic. |
| Structured-output prompts are not consistently strict. | `robot-arm` returns valid-looking JSON inside a Markdown fence. `forklift` returned raw parseable JSON in this run. | For examples that require machine parsing, prompt with "Return raw JSON only. Do not wrap in Markdown." In UI parsing, accept code-fenced JSON only as a compatibility fallback and flag it as non-strict. |
| Some examples are slow on the Transformers shim. | `forklift` took 114.0s and `mail-package` took 87.9s. | Keep latency metrics in the smoke runner. Upstream a durable vLLM path for the checkpoint, or document the Transformers backend as correctness-first rather than demo-latency-ready. |
| `mail-package` over-infers authorization. | The final answer says the person is likely allowed even though permission is not directly visible. | For safety/authorization examples, change the prompt to require "visible evidence only" and allow "cannot determine". Treat this as an inference quality issue. |

## Patch inventory already applied on Spark

These patches were applied to the live Spark instance during staging. They should
be moved into the owning repository before this becomes a durable release.

1. Vite streaming normalization in
   `apps/nvidia-build-reason-vite/server.mjs`: accumulate `reasoning` and
   `answer` channels, then emit a final normalized `openai` object as
   `chat.completion`.
2. Shared client normalization in `apps/_shared/reasonerClient.mjs`: normalize
   non-stream responses so `message.content` contains only the final answer and
   `message.reasoning_content` contains the trace.
3. Transformers/FastAPI backend shim at
   `/var/local/home/horde/cosmos3-edge/bin/cosmos3_edge_reasoner_backend.py`:
   serve `/v1/models` and `/v1/chat/completions`, split `</think>` output,
   stream reasoning and answer channels separately, and avoid an additional hard
   backend max-token cap unless explicitly configured. Note: the staging Vite
   process still forwarded the example `max_tokens=4096` request parameter.
4. Example asset repair on Spark: replaced Git LFS pointer files with the real
   `public/examples` and `dist/examples` assets.

## Exact output evidence

### robotics-next-action

![robotics-next-action first frame](assets/cosmos3-edge-reasoner-smoke/frames/robotics-next-action-first-frame.jpg)

Answer:

```text
Robot Arm 2 transports the USB Type C cable to the cardboard box and places it inside, possibly arranging it alongside the plastic tubes or within the air column wraps for cushioning.
```

Trace:

```text
Okay, let's see. The video is a lab or workshop with two robot arms. The main focus is on Robot Arm 2, which just picked up a USB-C cable. The setup includes a cardboard box with plastic tubes, air column wraps, and other items. The task seems to be packing or organizing.

Since Robot Arm 2 has just lifted the USB cable, the next logical step would be to move it to the cardboard box. The box is on the left, and the arm is on the right. So, the arm would need to transport the cable from the right side to the box on the left. After moving, it would place the cable into the box, maybe arranging it with the plastic tubes. Alternatively, maybe placing it into the air column wraps for cushioning. But since the air column wraps are already in the box, perhaps the next action is placing the cable into the box. The other robot arm (Arm 1) is stationary, so maybe Arm 2 is handling the main task here. So the immediate next action is moving the USB cable towards the box and placing it inside.
```

### robot-arm

![robot-arm first frame](assets/cosmos3-edge-reasoner-smoke/frames/robot-arm-first-frame.jpg)

Answer:

````text
```json
[
  {"point_2d": [410, 536], "label": "gripper trajectory"},
  {"point_2d": [410, 536], "label": "gripper trajectory"},
  {"point_2d": [410, 536], "label": "gripper trajectory"},
  {"point_2d": [410, 536], "label": "gripper trajectory"},
  {"point_2d": [413, 369], "label": "gripper trajectory"}
]
```
````

Trace:

```text
I will move my gripper from its current position at [410, 536] to the blue tape roll at [505, 721] to grasp it. After I have secured the tape roll, I will lift it and carry it to the open space between the roll and the basket at [495, 605], which gives me a clear path toward the target. Then I will carefully maneuver the tape roll into the gray woven basket at [460, 400], making sure it is fully placed inside. This sequence should give me a safe and precise transfer of the tape roll from its initial spot to the basket.

So my next steps should look like this: [410, 536], [410, 536], [410, 536], [410, 536], [413, 369]
```

### sdg-critic

![sdg-critic first frame](assets/cosmos3-edge-reasoner-smoke/frames/sdg-critic-first-frame.jpg)

Answer:

```text
Reject
```

Trace:

```text
I will move my gripper to [207, 100] to approach the blue block at [630, 290] and grasp it. Then I will lift the blue block from the yellow block at [560, 586] and reposition it to the left of the yellow block, making sure it is no longer on top. After placing the blue block, I will move my gripper to [495, 790] to interact with the red block, which is now supporting the yellow block. I will adjust the red block to ensure stability and prevent the yellow block from falling. Finally, I will move my gripper to [860, 925] to pick up the second blue block and place it on the table, completing the rearrangement.
```

### warehouse

![warehouse first frame](assets/cosmos3-edge-reasoner-smoke/frames/warehouse-first-frame.jpg)

Answer:

```text
Worker 2
```

Trace:

```text
Okay, let's see. The question is asking which worker picked up the dropped box. The video shows two workers: one in a yellow safety vest (Worker 0) and another in a plaid shirt (Worker 2).

First, I need to check the actions of each worker. Worker 0 is carrying boxes, drops one, and then walks away. The video shows that after dropping the box, he walks away. So he didn't pick it up again.

Then there's Worker 2. The video shows that after the dropped box, Worker 2 approaches the fallen box, bends down to pick it up, and then walks away. So clearly, Worker 2 is the one who picked up the dropped box.

The other workers shown are the one in the yellow vest (Worker 0) and the one in the plaid shirt (Worker 2). The video does not show any other workers. So the answer should be Worker 2.
```

### forklift

![forklift first frame](assets/cosmos3-edge-reasoner-smoke/frames/forklift-first-frame.jpg)

Answer:

```json
{
  "location": "The stack of bottled water on the pallet under the forklift",
  "estimated_weight": "150 lbs (or 100 lbs, depending on exact count)",
  "within_limit": true
}
```

Trace:

```text
Got it, let's tackle this problem step by step. First, we need to identify the load in the forklift. The load is a stack of bottled water, probably around 100 bottles or so, but let's estimate. Then, we need to check if the load is within the forklift's weight limit.

First, let's estimate the number of bottles. The stack looks like it's about 10 rows high, each row maybe 10 bottles, so 10*10=100 bottles. Each bottle is roughly 16 oz, so 100*16=1600 oz, which is about 100 pounds (since 16 oz = 0.5 lb, so 1600 oz / 16 = 100 lbs).

Next, forklift weight limits: typical forklifts can handle around 2,000-3,000 lbs, but we need to check if the load is within that. If the load is 100 lbs, that's well within the limit. But wait, maybe the load is more? Wait, the image shows a stack that's probably around 100-150 bottles. Let's say 150 bottles. 150*16=2400 oz, which is 150/16=9.375 lbs? Wait, no, wait. Wait, 16 oz per bottle, 150 bottles: 150*16=2400 oz. Convert to pounds: 2400 / 16 = 150 lbs. Wait, that's a lot. Wait, maybe I miscalculated. Wait, 16 oz per bottle, 100 bottles: 100*16=1600 oz = 100 lbs. 150 bottles: 150*16=2400 oz = 150 lbs. So if the forklift's maximum is, say, 2000 lbs, then 150 lbs is okay. But maybe the forklift's limit is higher. Wait, but the question is to estimate if it's in the limit.

Alternatively, maybe the load is a standard pallet of bottled water, which is often around 100-150 bottles. Let's say 150 bottles. Each bottle is about 16 oz, so total weight is 150*16=2400 oz = 150 lbs. If the forklift's maximum capacity is, say, 2000 lbs, then 150 lbs is within the limit. But maybe the forklift's limit is higher. Wait, but the problem says "estimate weights" and "if it's in the limit".

Wait, but maybe the load is more. Let's check the image again. The stack is a tall stack, maybe 10 rows high, each row has about 10 bottles, so 10*10=100 bottles. 100 bottles * 16 oz = 1600 oz = 100 lbs. If the forklift's maximum capacity is 2000 lbs, then 100 lbs is within the limit. So the estimated weight of the load is around 100-150 lbs, and if the forklift's limit is, say, 2000 lbs, then it's within the limit.

But the question is to return all as json: location, estimated weight, and if in limit. So first, identify the load's location (the stack of bottled water on the pallet), then estimate weight, then check if within limit.

So, location: the stack of bottled water on the pallet under the forklift. Estimated weight: let's say 150 lbs (or 100 lbs, depending on exact count). Then, if the forklift's limit is, for example, 2000 lbs, then it's within the limit.

Wait, but maybe the forklift's limit is higher. Let's assume the forklift can handle up to 2000 lbs. The load is around 150 lbs, so within limit.

So, putting it all together: location is the stack of bottled water on the pallet under the forklift. Estimated weight is around 150 lbs (or 100 lbs). If the forklift's limit is 2000 lbs, then it's within the limit.

But maybe the load is more. Let's say 150 bottles. 150*16=2400 oz = 150 lbs. So estimated weight is 150 lbs. If the forklift's maximum capacity is 2000 lbs, then it's within the limit.

So the JSON would be:

{
  "location": "The stack of bottled water on the pallet under the forklift",
  "estimated_weight": "150 lbs (or 100 lbs, depending on exact count)",
  "within_limit": true
}

Wait, but maybe the forklift's limit is higher. Let's check typical forklift capacities. Most forklifts can handle up to 2,000-3,000 lbs. So if the load is 150 lbs, it's well within. So the answer would be that the load is within the limit.
```

### mail-package

![mail-package first frame](assets/cosmos3-edge-reasoner-smoke/frames/mail-package-first-frame.jpg)

Answer:

```text
The person is likely allowed to pick up the packages. The video shows the man carrying a box and approaching the house with packages placed outside. There is no explicit indication of restrictions (e.g., signs, locked doors, or barriers) preventing him from taking the items. His casual attire and the presence of the packages on the mat suggest he is either the recipient or authorized to access them. The act of carrying a box implies he has already picked up the packages, further supporting the conclusion that he is permitted to do so.

**Answer:** Yes, the person is allowed to pick up the packages, as there are no shown restrictions, and his actions (carrying a box, approaching the door) imply he is the rightful owner or authorized to take them.
```

Trace:

```text
Okay, let's try to figure out if the person in the video is allowed to pick up the packages. The video shows a man approaching a house with packages outside. He's wearing casual clothes and a cap, and he's carrying a box. The house has a gray door with a black mat, and there are two cardboard boxes and a brown paper bag on the mat.

First, I need to consider the context. The man is holding a box and moving towards the door. The presence of the packages outside the house might indicate that they are either being delivered or are in the process of being delivered. The fact that the man is carrying a box suggests he might have just picked them up or is about to deliver them.

The question is whether he's allowed to pick them up. In many residential settings, when packages are left outside, it's common for the recipient to pick them up themselves. The man is dressed casually, which might suggest he's the delivery person or the owner. The video does not show any signs or barriers preventing him from taking the packages. The video doesn't show any security measures like locked doors or signs prohibiting access. The door is closed, but the man is approaching it, so maybe he's allowed to enter.

Also, the man is carrying a box, which implies he has already picked up the packages. If he's holding a box, he must have taken the packages from the location where they were placed. The presence of the packages on the mat suggests they are accessible. Unless there's a specific reason shown in the video that he's not allowed to take them, like a sign or a rule, the default assumption would be that he is allowed to pick them up.

However, the video doesn't show any signs or explicit permission. But since he's actively carrying a box and moving towards the door, it's logical to infer that he is the rightful owner or authorized to take the packages. Therefore, the answer is likely yes, he is allowed to pick them up.
```

## Reproduce

Run against a live Vite instance:

```bash
python tools/cosmos3_edge_build_smoke.py \
  --vite-origin http://127.0.0.1:5173 \
  --out /tmp/cosmos3-edge-build-smoke
```

The runner calls the Vite stream API, stores the media request source, extracts a
first-frame screenshot for each example, records `answer`, `reasoning`, and raw
SSE-normalized JSON, then flags Vite shape regressions and a small set of
model-compliance issues.
