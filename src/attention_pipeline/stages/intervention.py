"""Stage 6: structured event → LLM recommendation payload."""

from __future__ import annotations

import json
from typing import Any

from attention_pipeline.types import AttentionEvent, EMOTION_LABELS, POSTURE_LABELS


SYSTEM_PROMPT = """You are an instructional coach sitting beside a teacher.
You receive a structured attention-drop event from a classroom vision system.
Give three short fields:
1) explanation — what the signals suggest, without naming the student publicly
2) teaching_recommendation — one in-the-moment pedagogical move
3) intervention_suggestion — whether to wait, prompt the class, or check privately
Do not invent identity, grades, or medical claims. Stay under 120 words total.
"""


def event_to_prompt(event: AttentionEvent, class_mean_score: float | None = None, cue: str | None = None) -> str:
    payload = {
        "student_slot": event.student_id,
        "timestamp_s": round(event.timestamp, 2),
        "score": round(event.score, 1),
        "drop_kind": event.kind,
        "dominant_modality": event.dominant_modality,
        "behavioral_cue": cue,
        "features": event.explanation_features,
        "class_mean_score": class_mean_score,
        "emotion_labels": list(EMOTION_LABELS),
        "posture_labels": list(POSTURE_LABELS),
    }
    return (
        SYSTEM_PROMPT
        + "\n\nEvent JSON:\n"
        + json.dumps(payload, indent=2)
    )


def recommend(event: AttentionEvent, class_mean_score: float | None = None, cue: str | None = None) -> dict[str, Any]:
    """Context-aware pedagogical intervention engine providing specific instructional guidance."""
    modality = event.dominant_modality.replace("_", " ")
    cue_lower = (cue or "").lower()
    wait = event.kind == "sudden" and event.score > 30

    # 1. Device Distraction (Phone usage)
    if "phone" in cue_lower:
        explanation = (
            f"Persistent mobile device interaction detected ({event.kind}) "
            f"at engagement score {event.score:.0f}/100."
        )
        recommendation = (
            "Apply non-verbal proximity control: circulate calmly down the aisle near the flagged desk "
            "without halting instructional delivery."
        )
        suggestion = (
            "Avoid public confrontation; private physical presence typically causes the student to stow the device naturally."
        )
    # 2. Peer Distraction (Side talking / Turning to neighbor)
    elif "talk" in cue_lower or "disturb" in cue_lower:
        explanation = (
            f"Lateral social interaction or peer conversation detected ({event.kind}) "
            f"at engagement score {event.score:.0f}/100."
        )
        recommendation = (
            "Pivot immediately into a structured 30-second Think-Pair-Share check to productively channel "
            "conversational energy into academic discussion."
        )
        suggestion = (
            "Re-anchor the classroom with a quick concept question rather than singling out individual conversationalists."
        )
    # 3. Severe Fatigue / Sleep / Head Collapsed on Table
    elif "sleep" in cue_lower or "table" in cue_lower:
        explanation = (
            f"Postural collapse or persistent downward orientation ({event.kind}) "
            f"indicating cognitive exhaustion at score {event.score:.0f}/100."
        )
        recommendation = (
            "Trigger a whole-class physical or conceptual energy reset: have all students stand for a quick stretch "
            "or conduct a rapid live poll."
        )
        suggestion = (
            "Discreetly check in with the student during the subsequent transition to screen for acute fatigue or wellness issues."
        )
    # 4. Visual Attention Drift (Looking away / Gaze drift)
    elif "left" in cue_lower or "right" in cue_lower or "away" in cue_lower or modality in ("gaze", "head pose"):
        explanation = (
            f"Visual attention drifted from instructional zone ({event.kind}) with {modality} "
            f"at score {event.score:.0f}/100."
        )
        recommendation = (
            "Direct an inclusive, low-stakes retrieval question toward that sector of the room, "
            "or use a vocal inflection change to refocus eyes on the visual aid."
        )
        suggestion = (
            "Wait one brief observation cycle; isolated look-aways are normal cognitive breathers."
            if wait
            else "Circulate toward that side of the room and re-anchor visual engagement."
        )
    # 5. Default General Fallback
    else:
        explanation = (
            f"Attention dropped ({event.kind}) with {modality} as the strongest cue "
            f"at score {event.score:.0f}/100."
        )
        recommendation = (
            "Pause for a 15-second retrieval prompt aimed at the whole class, "
            "then cold-call a nearby volunteer rather than the flagged seat."
        )
        suggestion = (
            "Wait one more cycle; brief dips are common."
            if wait
            else "Circulate toward that side of the room and use a quiet check-in."
        )

    return {
        "explanation": explanation,
        "teaching_recommendation": recommendation,
        "intervention_suggestion": suggestion,
        "prompt": event_to_prompt(event, class_mean_score, cue=cue),
    }
