from openai_harmony import (
    Role,
    Message,
    Conversation,
    SystemContent,
    ReasoningEffort,
    load_harmony_encoding,
    HarmonyEncodingName,
)


def gpt_oss_high_reasoning(user_instruction: str) -> str:
    """
    example output format with assistant prefill:

    <|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
    Knowledge cutoff: 2024-06

    Reasoning: high

    # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>USER_INSTR_HERE<|end|><|start|>assistant
    """
    # gpt-oss defaults to medium effort https://huggingface.co/openai/gpt-oss-120b/blob/main/chat_template.jinja#L6
    # chain in with_model_identity() for role-playing
    system = Message.from_role_and_content(
        Role.SYSTEM, SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH)
    )
    user = Message.from_role_and_content(Role.USER, user_instruction)
    convo = Conversation.from_messages([system, user])
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    # alternatively: if we want to prefill the assistant response
    # toks = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
    toks = enc.render_conversation(convo)
    return enc.decode_utf8(toks)
