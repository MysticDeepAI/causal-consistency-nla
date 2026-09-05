"""
Stage 3 -- Claims judge (kappa and nu operators, Def. 2.4 of the framework doc).

Extracts atomic claims from an AV explanation and attributes each to a
domain variable. GOLDEN RULE: the judge extracts what the explanation
ASSERTS, not what the source text says -- confabulations are data, not
errors.

Freezing protocol (must be run once before trusting any downstream number):
 1. Hand-label N=30 explanations (config: judge.n_validation_explanations).
 2. Measure judge-human agreement on (a) variable attribution, (b) value.
 3. If both >= judge.agreement_threshold: freeze this prompt (git-tag the
    commit) and report the figure in the write-up.
    If not: iterate the few-shots (never the schema) and re-validate on
    30 NEW explanations.

Output: results/results_stage3.json
  { "pair_id": {"factual": [[claim, ...], ...K],
                "counterfactual": [[claim, ...], ...K]}, ... }
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

VALID_VARS = {"X", "W", "Z", "Y", "THEME"}

JUDGE_PROMPT = """You are annotating explanations produced by an Activation \
Verbalizer (AV) -- a system that describes what a language model's internal \
activation encodes. Your job: extract every atomic claim the explanation \
makes and attribute each to a domain variable.

DOMAIN VARIABLES (a street scene with one pedestrian):
- X = WEATHER. value 1 = rainy, 0 = sunny.
- W = FOOTWEAR. value 1 = boots, 0 = sneakers.
- Z = PACE. value 1 = hurried pace, 0 = normal pace.
- Y = COFFEE STOP. value 1 = likely to stop for coffee, 0 = unlikely.
- THEME = claims about format, genre, language, document type, or
  next-token expectations that do NOT assert a value of X/W/Z/Y.

CRITICAL RULES:
1. Extract what the explanation ASSERTS, even if you suspect it is
   confabulated or contradicts itself. You never see the original text; do
   not guess it.
2. One claim = one variable = at most one asserted value. Split compound
   sentences into atomic claims.
3. "value" is the asserted value mapped to 0/1. If the explanation mentions
   the variable without committing to a value (e.g., "expects a footwear
   descriptor next"), set value=null and specificity="presence".
4. Predictions about the NEXT token that assert a value ("likely continues
   with 'boots'") count as value claims with confidence="low".
5. Output ONLY the JSON object, no prose.

OUTPUT SCHEMA:
{"claims": [{"text": "...", "variable": "X|W|Z|Y|THEME",
             "value": "0|1|null", "specificity": "value|presence",
             "confidence": "high|low"}]}

=== EXAMPLE 1 ===
Explanation:
"Street-scene diary format with an observational tone suggests a pedestrian
log. The phrase 'wearing boots on a rainy afternoon' strongly implies a
weather-appropriate footwear descriptor. Final token is mid-clause, part of
a pedestrian description, immediately requiring a phrase like 'moving at a
hurried clip' completing the pace descriptor."

Output:
{"claims": [
 {"text": "street-scene diary, observational log", "variable": "THEME", "value": null, "specificity": "presence", "confidence": "high"},
 {"text": "asserts boots", "variable": "W", "value": "1", "specificity": "value", "confidence": "high"},
 {"text": "asserts rainy afternoon", "variable": "X", "value": "1", "specificity": "value", "confidence": "high"},
 {"text": "expects hurried pace continuation", "variable": "Z", "value": "1", "specificity": "value", "confidence": "low"}
]}

=== EXAMPLE 2 (contradicts the source on purpose -- still extract as asserted) ===
Explanation:
"The pedestrian is described in sneakers, moving briskly despite the sunny
weather, likely heading somewhere rather than stopping for coffee."

Output:
{"claims": [
 {"text": "asserts sneakers", "variable": "W", "value": "0", "specificity": "value", "confidence": "high"},
 {"text": "asserts hurried movement", "variable": "Z", "value": "1", "specificity": "value", "confidence": "high"},
 {"text": "asserts sunny weather", "variable": "X", "value": "0", "specificity": "value", "confidence": "high"},
 {"text": "asserts unlikely to stop for coffee", "variable": "Y", "value": "0", "specificity": "value", "confidence": "high"}
]}

=== EXAMPLE 3 (no domain content) ===
Explanation:
"Field-note register with terse, clipped sentence structure typical of an
observation log; the truncated final clause expects a short factual
continuation."

Output:
{"claims": [
 {"text": "field-note register, observation log", "variable": "THEME", "value": null, "specificity": "presence", "confidence": "high"},
 {"text": "expects short factual continuation", "variable": "THEME", "value": null, "specificity": "presence", "confidence": "high"}
]}

=== NOW ANNOTATE ===
Explanation:
"{explanation}"

Output:"""


def prompt_hash() -> str:
    """Hash used to freeze the prompt in git and cite it in the write-up."""
    return hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()[:12]


def parse_judge_output(raw: str) -> list:
    """Defensive parsing of the judge's output."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1].removeprefix("json").strip()
    obj = json.loads(s)
    claims = []
    for c in obj.get("claims", []):
        if c.get("variable") in VALID_VARS:
            v = c.get("value")
            c["value"] = None if v in (None, "null") else int(v)
            claims.append(c)
    return claims


def qhat(claims_per_rollout: list, variable: str) -> dict:
    """
    Empirical distribution q_hat_V (Def. 2.4) over {0, 1, BOT} from K
    rollouts. BOT = the rollout makes no value-claim about V.
    Per-rollout aggregation: majority value among that rollout's
    value-claims on V (a tie or absence -> BOT).
    """
    outcomes = []
    for claims in claims_per_rollout:
        vals = [c["value"] for c in claims
                if c["variable"] == variable and c["value"] is not None]
        if not vals:
            outcomes.append("BOT")
        else:
            cnt = Counter(vals)
            top = cnt.most_common()
            outcomes.append("BOT" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0])
    K = len(outcomes)
    cnt = Counter(outcomes)
    return {k: cnt.get(k, 0) / K for k in [0, 1, "BOT"]}


import re

WEATHER_RAINY_PATTERN = re.compile(
    r"\b(rainy|raining|rainfall|downpour|drizzl\w*|stormy|storm)\b", re.IGNORECASE)
WEATHER_SUNNY_PATTERN = re.compile(
    r"\b(sunny|sunshine|sunlit|clear sky|clear skies)\b", re.IGNORECASE)


def keyword_classify_weather(text: str) -> list:
    """
    Code-only classifier for X (weather) -- needed for the do(W) mirror
    experiment, where X (not W) is the variable under the invariance test.

    Uses WORD-BOUNDARY regex, not bare substring matching, and deliberately
    does NOT include a bare "rain" token: this project's confabulations
    frequently mention "raincoat"/"rain gear"/"rain boots" as clothing
    items regardless of the text's actual declared weather value, so a
    substring match on "rain" would misfire on those. Broadened beyond the
    single word "rainy"/"sunny" (v1) after finding those literal words
    were rarely used by the AV even when weather was clearly being
    described (storms, drizzle, sunshine, etc. instead).
    """
    has_rainy = bool(WEATHER_RAINY_PATTERN.search(text))
    has_sunny = bool(WEATHER_SUNNY_PATTERN.search(text))
    if has_rainy and not has_sunny:
        value = 1
    elif has_sunny and not has_rainy:
        value = 0
    else:
        value = None
    return [{"text": text[:80], "variable": "X", "value": value,
            "specificity": "value", "confidence": "high"}]


def keyword_classify_footwear(text: str) -> dict:
    """
    Cheap, code-only classifier for W (footwear) -- no LLM call needed.
    W is a clean two-word binary variable ("boots" vs "sneakers"), unlike
    the general free-text claims the JUDGE_PROMPT above is designed for.
    Returns a single-claim list in the same schema qhat() expects.

    KNOWN LIMITATIONS (spot-check a sample before trusting at scale):
      - misses negation ("not wearing boots" is misclassified as boots=1)
      - "both" mentioned (rare, e.g. quoting the context block verbatim)
        is treated as no-claim (BOT), not as a separate category
      - does not distinguish "boots" as a claim about THIS pedestrian
        from an incidental mention elsewhere in a confabulated tangent
    """
    t = text.lower()
    has_boot = "boot" in t
    has_sneaker = "sneaker" in t
    if has_boot and not has_sneaker:
        value = 1
    elif has_sneaker and not has_boot:
        value = 0
    else:
        value = None  # neither, or both (ambiguous) -> no usable claim
    return [{"text": text[:80], "variable": "W", "value": value,
            "specificity": "value", "confidence": "high"}]


# ---------------- Descendant sensitivity: Z (pace), Y (coffee-stop) ----------------
# Positive-control classifiers. Z and Y are CAUSAL DESCENDANTS of X (weather) in
# the SCM, so unlike W, their claims SHOULD shift under do(X). If they don't
# shift either, the AV isn't respecting causal structure selectively -- it's
# just insensitive to the intervention altogether, which would undercut the W
# result (a "dead" AV trivially passes the invariance test for the wrong reason).

PACE_HURRIED_WORDS = ["hurried", "brisk", "quick", "quickly", "rushing",
                      "rushed", "hastily", "fast pace", "fast-paced",
                      "fast walk", "walking fast", "speeding", "swift",
                      "energetic pace", "rapid pace", "hurrying", "in a hurry",
                      "walking quickly", "quick step", "brisk pace",
                      "accelerated", "sprinting", "walkiness: high"]
PACE_NORMAL_WORDS = ["normal pace", "casual", "relaxed", "leisurely",
                     "unhurried", "steady pace", "slow", "ambling",
                     "slow pace", "walking slowly", "calm pace", "strolling",
                     "gentle pace", "moderate pace", "easy pace",
                     "no rush", "not in a hurry", "walkiness: low"]


def keyword_classify_pace(text: str) -> list:
    """Code-only classifier for Z (pace). Same limitations as footwear's
    classifier (no negation handling); word lists are not exhaustive."""
    t = text.lower()
    has_hurried = any(w in t for w in PACE_HURRIED_WORDS)
    has_normal = any(w in t for w in PACE_NORMAL_WORDS)
    if has_hurried and not has_normal:
        value = 1
    elif has_normal and not has_hurried:
        value = 0
    else:
        value = None
    return [{"text": text[:80], "variable": "Z", "value": value,
            "specificity": "value", "confidence": "high"}]


COFFEE_YES_PHRASES = ["stop for coffee", "stopping for coffee", "stops for coffee",
                      "grab a coffee", "grabbing a coffee", "gets coffee",
                      "getting coffee", "coffee break", "likely to stop",
                      "will stop for coffee", "heads to a cafe", "heading to a cafe",
                      "stops at a cafe", "visits a coffee shop", "grabs a latte",
                      "coffee-stop likelihood: high", "coffee stop: yes",
                      "buys coffee", "picks up coffee", "coffee run"]
COFFEE_NO_PHRASES = ["skip coffee", "skips coffee", "skipping coffee", "no coffee",
                     "won't stop", "will not stop", "not stop for coffee",
                     "unlikely to stop", "doesn't stop for coffee",
                     "does not stop for coffee", "avoids coffee",
                     "passes the coffee shop", "walks past the cafe",
                     "no time for coffee", "coffee-stop likelihood: low",
                     "coffee stop: no", "skips the cafe", "bypasses the coffee shop"]


def keyword_classify_coffee(text: str) -> list:
    """
    Code-only classifier for Y (coffee-stop). Phrase-based (not a bare
    "coffee" keyword) specifically to handle the yes/no distinction, since
    both classes mention the word "coffee" -- a bare-keyword approach
    (like footwear's) can't distinguish them.

    BUG FOUND AND FIXED: negation phrases like "unlikely to stop" contain
    "stop for coffee" as a literal substring ("...unlikely to STOP FOR
    COFFEE today"), so both YES and NO phrase lists matched simultaneously
    and collapsed to no-claim. Fix: check NO phrases first and let them
    take precedence -- they're the more specific/restrictive match.
    """
    t = text.lower()
    has_no = any(p in t for p in COFFEE_NO_PHRASES)
    has_yes = any(p in t for p in COFFEE_YES_PHRASES)
    if has_no:
        value = 0
    elif has_yes:
        value = 1
    else:
        value = None
    return [{"text": text[:80], "variable": "Y", "value": value,
            "specificity": "value", "confidence": "high"}]


def keyword_classify_all(text: str) -> list:
    """Combined classifier: one judging pass yields claims about X, W, Z,
    and Y from the same explanation text -- reusable for both the do(X)
    experiment (H1 test on W) and the do(W) mirror experiment (H1' test
    on X)."""
    return (keyword_classify_weather(text)
           + keyword_classify_footwear(text)
           + keyword_classify_pace(text)
           + keyword_classify_coffee(text))


def call_judge(explanation: str, backend: str = "claude_api") -> list:
    """
    Dispatches to the configured judge backend.
      - "keyword_footwear": W only, no API/model call.
      - "keyword_all": W + Z + Y in one pass, no API/model call.
      - "claude_api" / "qwen_local": general free-text claims judge using
        JUDGE_PROMPT -- plug in the project's LLM-call harness here for
        open-ended multi-variable claim extraction beyond keyword matching.
    """
    if backend == "keyword_footwear":
        return keyword_classify_footwear(explanation)
    if backend == "keyword_all":
        return keyword_classify_all(explanation)
    raise NotImplementedError(
        "Wire this to the project's existing API/local-model call helper."
    )


def judge_items(items, config):
    """Entry point for this stage. items: output of verbalization.verbalize()
    -- each item has item["explanations"] = {"factual": [K texts],
    "counterfactual": [K texts]}. Returns items with a new "claims" field:
    {"factual": [[claim,...], ...K], "counterfactual": [...]}."""
    backend = config["judge"]["backend"]
    for i, item in enumerate(items, 1):
        item["claims"] = {
            "factual": [call_judge(e, backend) for e in item["explanations"]["factual"]],
            "counterfactual": [call_judge(e, backend) for e in item["explanations"]["counterfactual"]],
        }
        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] items judged")
    print(f"[judging] prompt_hash={prompt_hash()} done: {len(items)} items")
    return items


if __name__ == "__main__":
    print("Judge prompt hash:", prompt_hash())
    demo = ('{"claims": [{"text": "boots", "variable": "W", "value": "1", '
            '"specificity": "value", "confidence": "high"}]}')
    print("Parse demo:", parse_judge_output(demo))
