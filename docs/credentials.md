# File-only API authentication

Version 0.2 reads API keys exclusively from explicitly configured local files.
The environment-variable authentication used by version 0.1 is not retained as
a fallback. This makes the credential source unambiguous for every experiment.

The standard files are `secrets/openai.key`, `secrets/anthropic.key`,
`secrets/google.key` and `secrets/openrouter.key`. Paste one key into each file
that is needed. An unused provider's file can remain empty. The exact path is
selected by the model catalog's `api_key_file` field. A custom endpoint may use
another file. Relative paths are resolved from the repository root and are frozen
as locations, not contents.

## When keys are read

Importing code, listing models, planning runs, calculating costs and reporting
completed experiments do not read credential files. A credential preflight
explicitly validates the selected files. The provider reads its key immediately
before making an authenticated HTTP operation. No key is copied into the model
configuration, request body, prompt, generation record, judgement or report.

Missing, unreadable, empty, multiline, quoted, JSON-formatted and assignment-style
key files produce a safe diagnostic identifying only the path and required format.
No diagnostic prints the contents, length or a fingerprint of a key. A trailing
newline, including a Windows line ending, is accepted. Save files without a UTF-8
byte-order mark.

An explicitly configured localhost endpoint can opt into unauthenticated access.
That exception cannot be used with a remote hostname. An explicitly configured
key file is never bypassed because it is empty or invalid.

## Git and publication

The `secrets/` directory is ignored except for its README. `*.key`, `*.pem` and
`.env` files are also ignored. Empty placeholders in the download are convenient
local setup files and will not be committed by an ordinary `git add`.

Ignore rules protect untracked files. They do not erase prior commits or prevent
an explicit force-add. Previously committed live credentials must be removed
from tracking and rotated. Manual archives require the same care as Git. The
release packaging workflow must never include live contents from `secrets/`.

Authentication headers exist transiently in memory for the provider request.
They are not part of the experiment journal. If a provider echoes a loaded key
in an error or response body, the logging boundary replaces it with
`[REDACTED]`. Credential-named fields in parsed provider diagnostics are also
masked. The raw provider record is otherwise retained, with any necessary
credential redaction taking precedence over byte-for-byte retention.
