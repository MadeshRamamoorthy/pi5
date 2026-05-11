"""Copy library for the kiosk.

Ported verbatim from the client's `ai_lab.txt`. Each category is a list
so we can pick a random variant on every utterance — keeps the kiosk
from sounding canned.

Helpers:
    random_error("face_recognition_failed") -> str
    random_recognized_greeting(name)        -> str
    random_unrecognized_greeting()          -> str
    random_registration_prompt("ask_name")  -> str
    get_rotating_content()                  -> {"type", "icon", "content"}
"""

from __future__ import annotations

import random


ERROR_MESSAGES = {
    "face_recognition_failed": [
        "Hmm, the lighting's a bit tricky. Mind moving slightly left?",
        "I'm having trouble seeing you clearly. Can you adjust your position?",
        "The lighting's playing tricks on me. Try moving a bit to the right?",
    ],
    "microphone_no_input": [
        "Oops, didn't quite catch that! Try again or just type your question on screen?",
        "Sorry, I missed that. Want to try speaking again or type instead?",
        "Hmm, I didn't hear anything. Mind repeating that?",
    ],
    "cannot_answer": [
        "Great question! I'm still learning that one. Want me to connect you with someone who knows? Email Calgary_AIClub@infosys.com",
        "That's a tough one! I'm not sure yet. Reach out to Calgary_AIClub@infosys.com for expert help!",
        "Interesting question! That's beyond my knowledge right now. Try emailing Calgary_AIClub@infosys.com",
    ],
    "multiple_faces_detected": [
        "Whoa, a crowd! I see multiple people. Who wants to chat first? Wave again!",
        "I see a few of you! One at a time, please. Who's going first?",
    ],
    "camera_error": [
        "Oops! My camera seems to be having a moment. Let me try that again...",
        "Camera hiccup! Give me just a second to reset...",
    ],
}


RECOGNIZED_GREETINGS = [
    "{name}! My favorite human! Ask me anything!",
    "Welcome, {name}! Ready to explore some AI?",
    "Hey {name}! Great to see you. What can I help with?",
]


UNRECOGNIZED_GREETINGS = [
    "Hey there! I don't think we've met. I'm ECHO SCOPE. Want to introduce yourself so I remember you next time?",
    "Ooh, a new face! I'm ECHO SCOPE. Want me to remember you for next time? Let's do a quick intro!",
    "Hi! I don't recognize your face yet. Mind if I get to know you? It just takes 10 seconds!",
    "Hello! I'm ECHO SCOPE, and I don't think we've met. Want me to remember you?",
]


REGISTRATION_PROMPTS = {
    "ask_name": [
        "Great! What's your name?",
        "Awesome! What should I call you?",
        "Perfect! Tell me your name.",
    ],
    "confirm_registration": [
        "Got it! I'll remember you, {name}. Welcome to ECHO AI space!",
        "Perfect! {name} — you're all set. Great to meet you!",
        "Awesome, {name}! I'll recognize you next time. Welcome!",
    ],
    "decline_registration": [
        "No worries! You can still ask me anything. What's on your mind?",
        "That's totally fine! How can I help you today?",
    ],
}


AI_FUN_FACTS = [
    "The term 'Artificial Intelligence' was coined in 1956 at the Dartmouth Conference!",
    "ChatGPT reached 100 million users in just 2 months — the fastest-growing app ever!",
    "AI can now detect diseases from medical images with accuracy matching human doctors!",
    "The first AI program was created in 1951 — a checkers-playing program called 'Turochamp'!",
    "By 2025, the global AI market is expected to reach $190 billion!",
    "AI models like GPT-4 were trained on data from 570 GB of text — that's millions of books!",
    "Deep learning neural networks are inspired by how human brains process information!",
    "AI can now generate realistic images from text descriptions in seconds — it's called text-to-image AI!",
    "Self-driving cars use AI to process data from cameras, radar, and lidar sensors simultaneously!",
    "AI language models can now understand and generate text in over 100 languages!",
    "The largest AI models today have over 1 trillion parameters — more than stars in the Milky Way!",
    "AI can compose music, write poetry, and create art — blurring the line between human and machine creativity!",
    "Machine learning algorithms improve automatically through experience without being explicitly programmed!",
    "AI assistants process millions of requests every day, learning to understand context and nuance!",
    "Reinforcement learning is how AI masters games like Chess and Go — by playing against itself millions of times!",
]


AI_TIPS = [
    "When prompting AI: be specific! 'Write an email' vs 'Write a professional follow-up email to a client'",
    "Use 'Act as a [role]' to get better AI responses — try 'Act as a data analyst and explain this...'",
    "Break complex tasks into steps — AI performs better with clear, sequential instructions!",
    "Always fact-check AI outputs — they can confidently provide incorrect information (hallucinations)!",
    "Give AI examples! Show what you want: 'Like this: [example]. Now do it for: [your task]'",
    "Iterate and refine — your first prompt rarely gives the best result. Ask AI to improve its own output!",
    "Context matters! Provide background information to help AI understand what you really need.",
    "Use AI for brainstorming — ask for '10 different approaches to...' and explore creative options!",
    "Specify format: 'Give me a bulleted list' or 'Write in 3 paragraphs' for structured outputs.",
    "Ask AI to explain its reasoning — add 'Explain your thought process step-by-step' to prompts!",
    "Use AI as a learning tool — ask 'Explain [concept] like I'm a beginner' for complex topics!",
    "Combine tools! Use AI for drafting, then refine with human creativity and expertise.",
]


# --------------- helpers -----------------------------------------------------


def _pick(items: list[str]) -> str:
    return random.choice(items) if items else ""


def random_error(category: str) -> str:
    return _pick(ERROR_MESSAGES.get(category, []))


def random_recognized_greeting(name: str) -> str:
    return _pick(RECOGNIZED_GREETINGS).format(name=name)


def random_unrecognized_greeting() -> str:
    return _pick(UNRECOGNIZED_GREETINGS)


def random_registration_prompt(category: str, **fmt) -> str:
    line = _pick(REGISTRATION_PROMPTS.get(category, []))
    return line.format(**fmt) if fmt else line


def get_rotating_content() -> dict:
    """Alternates between a fact and a tip, matching the client's
    `{type, icon, content}` schema."""
    if random.random() < 0.5:
        return {"type": "AI FUN FACT", "icon": "💡",
                "content": _pick(AI_FUN_FACTS)}
    return {"type": "AI TIP", "icon": "📰", "content": _pick(AI_TIPS)}
