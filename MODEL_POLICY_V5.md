# Model Policy (Hybrid v5)

Date: 2026-02-24
Primary script: `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py`

## Scope
This document describes how post-processing model configuration works in Hybrid v5 for both **Refine Text** and **Promptify** actions.

## Explicit first-class models
Only the following models are explicitly configured:

- `phi4:latest`
- `glm-4.7-flash:latest`

These are the only hardcoded default entries in `ModelConfig.get_default_configs()`.

## Dynamic routing for all other models
Any model not listed above is treated as an unknown model and routed dynamically by `ModelConfig.get_config()`.

### Routing rules
- If model is detected as GPT-OSS family:
  - Use GLM-style profile (`refine_prompt_key='glm_47_flash'`)
  - Force `think='high'`

- Else if model is detected as thinking-capable:
  - Use GLM-style profile (`refine_prompt_key='glm_47_flash'`)
  - Use `think=True`

- Else (non-thinking or unknown capabilities):
  - Use Phi-style profile (`refine_prompt_key='phi4'`)
  - Use `think=None`

### Metadata and fallback behavior
- The app tries `ollama show` metadata when resolving unknown models.
- GPT-OSS detection also works by model name/family tokens even if `ollama show` metadata is unavailable.
- For non-GPT unknown models without metadata, the fallback remains Phi-style.

## Prompt catalogs
Prompt routing remains catalog-based:

- Refine catalogs:
  - `phi4`
  - `glm_47_flash`
- Promptify catalog:
  - `default`

## UI defaults
Model list behavior in the selector:

- If model list can be fetched from Ollama, use installed models.
- If model list fetch fails/empty, fallback list is:
  - `phi4:latest`
  - `glm-4.7-flash:latest`
- Selector prefers `phi4:latest` when present.

## Operational note
This policy does **not** remove any models from Ollama. It only changes runtime model-selection logic in code.
