summarization_prompt = """The following is a transcript from an educational video on nature.
Summarize the transcript, extracting the main takeaways.
Focus on curious facts. Use a friendly tone and write in simple English.

Video title: {title}
Video transcript:
---
{transcript}
---

Summarization of the video transcript in 100 words or less:
"""


def _cap_prompt(
    tokenize,
    detokenize,
    prompt_template,
    prompt_params,
    max_length_input,
):
    prompt = prompt_template.format(**prompt_params)
    num_prompt_tokens = len(tokenize(prompt))
    capped_prompt = prompt
    if num_prompt_tokens > max_length_input:
        # Prompt won't fit, reduce transcript and recreate prompt
        transcript_tokens = tokenize(prompt_params["transcript"])
        template_length = len(tokenize(prompt_template))
        max_transcript_tokens = max_length_input - template_length
        capped_transcript = detokenize(transcript_tokens[:max_transcript_tokens])
        capped_prompt_params = prompt_params
        capped_prompt_params["transcript"] = capped_transcript
        capped_prompt = prompt_template.format(**capped_prompt_params)
        print(
            f"Limiting transcript from {len(transcript_tokens)} to {max_transcript_tokens} tokens to fit context window."
        )

    return capped_prompt


def get_summarization_prompt(
    tokenize,
    detokenize,
    title,
    transcript,
    max_length_input,
):
    prompt_params = {"title": title, "transcript": transcript}
    return _cap_prompt(
        tokenize, detokenize, summarization_prompt, prompt_params, max_length_input
    )
