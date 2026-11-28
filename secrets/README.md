# Local API key files

Paste each API key into the corresponding local file, as one plain-text line.
Do not add quotes, JSON, an environment-variable assignment or comments.

| File | Service |
|---|---|
| `openai.key` | OpenAI |
| `anthropic.key` | Anthropic |
| `google.key` | Gemini Developer API |
| `openrouter.key` | OpenRouter and its explicitly configured model providers |

The downloaded archive includes empty placeholder files. They contain no keys.
Only this README is tracked by Git. The actual key files are ignored by the
repository's `.gitignore`, including when the repository is pushed to GitHub.

If the repository was cloned from GitHub, create the four files yourself. Use
UTF-8 text, without a byte-order mark. On Linux/macOS, restricting permissions to
the owner is appropriate, for example `chmod 600 secrets/*.key`.

The model catalog names the path using `api_key_file`. Relative paths resolve
from the repository root, not the directory from which a command happens to run.
Additional explicitly configured providers can use another file in this folder.

The harness reads credentials only from these files. Environment variables do not
provide fallback credentials. Key values are never copied to a frozen run,
prompt, result export or report. Planning and cost estimation do not read them.

Git ignore rules do not remove a file that was already committed. If an actual
key was previously committed, remove it from tracking and rotate that key. Do
not use `git add -f` on this directory. When sharing a ZIP manually, use the
release packaging helper, which excludes keys or replaces placeholders with
empty bytes, rather than zipping a working repository containing live keys.
