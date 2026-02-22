import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import tiktoken


PATH_CHATs = "../../ChatGPTChats"
ENC = tiktoken.encoding_for_model("gpt-5")
PRINCING_1M_TOKEN = {
    "openai_gpt_5_2": {
        "input": 1.75,
        "output": 14.00,
    },
    "openai_gpt_5_2_pro": {
        "input": 21.00,
        "output": 168.00,
    },
    "anthropic_claude_sonnet_4_6": {
        "input": 3.00,
        "output": 15.00,
    },
    "anthropic_claude_opus_4_6": {
        "input": 5.00,
        "output": 25.00,
    },
    "google_gemini_3_pro_preview": {
        "input": 2.00,
        "output": 12.00,
    },
}


def _to_dt(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def load_conversations(file_path):
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        print(
            f"Number of conversations loaded successfully: {len(data)}"
        )
        return data
    raise ValueError(
        f"Expected a list of conversations, got {type(data)}"
    )


def _extract_text_from_message(message: dict[str, any]) -> str:
    content = message.get("content", {})
    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        parts = content.get("parts", [])
        if isinstance(parts, list):
            out = []
            for p in parts:
                if isinstance(p, dict):
                    out.append(
                        p.get("text", "") or p.get("content", "") or ""
                    )
                else:
                    out.append(str(p))
            return "\n".join([s for s in out if s])
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        return "\n".join([str(c) for c in content if isinstance(c, str)])

    return ""


def linearize_conversation(conv: dict[str, any]) -> list[dict[str, any]]:
    mapping = conv.get("mapping", {})
    current_node = conv.get("current_node")

    if not current_node and mapping:
        current_node = next(reversed(mapping.keys()))

    node_ids: list[str] = []
    nid = current_node
    while nid:
        node_ids.append(nid)
        node = mapping.get(nid, {})
        nid = node.get("parent")

    node_ids.reverse()
    msgs: list[dict[str, any]] = []
    for nid in node_ids:
        node = mapping.get(nid, {})
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role")
        text = _extract_text_from_message(msg).strip()
        if not role:
            continue
        msgs.append(
            {
                "node_id": nid,
                "role": role,
                "text": text,
                "create_time": _to_dt(msg.get("create_time")),
            }
        )
    return msgs


def count_tokens(message: str) -> int:
    return len(ENC.encode(message))


def extract_year_and_month(dt_str: Optional[str]) -> Optional[str]:
    if dt_str is None:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m")
    except ValueError:
        print(f"Invalid date format: {dt_str}")
        return None


def main(debug=False):
    conversations_path = os.path.join(PATH_CHATs, "conversations.json")
    if not os.path.exists(conversations_path):
        raise FileNotFoundError(f"File not found: {conversations_path}")
    conversations = load_conversations(conversations_path)
    conversations_by_date = {}
    for i, conv in enumerate(conversations):
        title = conv.get("title", "No Title")
        conv_id = conv.get("id", "No ID")
        created = _to_dt(conv.get("create_time"))
        msgs = linearize_conversation(conv)
        if debug:
            for msg in msgs:
                if msg["role"] != "system" and msg["text"]:
                    print(f"{msg['role']}: {msg['text'][:100]}...")
                    print(f"Tokens: {count_tokens(msg['text'])}")
                    print("create time:" + msg["create_time"])
        timekey = extract_year_and_month(created)
        conversations_by_date.setdefault(timekey, []).append(
            {
                "id": conv_id,
                "title": title,
                "created": created,
                "messages": msgs,
                "input_tokens": sum(
                    count_tokens(msg["text"])
                    for msg in msgs
                    if msg["role"] == "user"
                ),
                "output_tokens": sum(
                    count_tokens(msg["text"])
                    for msg in msgs
                    if msg["role"] == "assistant"
                ),
            }
        )

    # So, how much tokens do we have by month
    print("\n=== Tokens by month ===")
    for timekey, convs in conversations_by_date.items():
        total_input_tokens = sum(conv["input_tokens"] for conv in convs)
        total_output_tokens = sum(
            conv["output_tokens"] for conv in convs
        )
        print(
            f"{timekey}: {total_input_tokens} tokens across {len(convs)} conversations"
        )
        print(
            f"{timekey}: {total_output_tokens} tokens across {len(convs)} conversations"
        )
    max_input_tokens = max(
        sum(conv["input_tokens"] for conv in convs)
        for convs in conversations_by_date.values()
    )
    max_output_tokens = max(
        sum(conv["output_tokens"] for conv in convs)
        for convs in conversations_by_date.values()
    )
    avg_input_tokens = sum(
        sum(conv["input_tokens"] for conv in convs)
        for convs in conversations_by_date.values()
    ) / len(conversations_by_date)
    avg_output_tokens = sum(
        sum(conv["output_tokens"] for conv in convs)
        for convs in conversations_by_date.values()
    ) / len(conversations_by_date)

    print(f"max input tokens monthly usage: {max_input_tokens}")
    print(f"max output tokens monthly usage: {max_output_tokens}")
    print(f"average input tokens monthly usage: {avg_input_tokens}")
    print(f"average output tokens monthly usage: {avg_output_tokens}")

    print("\n =====Princing estimation=====")

    for model, pricing in PRINCING_1M_TOKEN.items():
        print(f"\n\n==== Model: {model} ======\n\n")
        max_input_cost = max_input_tokens * pricing["input"] / 1_000_000
        max_output_cost = (
            max_output_tokens * pricing["output"] / 1_000_000
        )
        avg_input_cost = avg_input_tokens * pricing["input"] / 1_000_000
        avg_output_cost = (
            avg_output_tokens * pricing["output"] / 1_000_000
        )

        print(f"Max input cost: ${max_input_cost:.2f}")
        print(f"Max output cost: ${max_output_cost:.2f}")
        print(f"Average input cost: ${avg_input_cost:.2f}")
        print(f"Average output cost: ${avg_output_cost:.2f}")


if __name__ == "__main__":
    main()
