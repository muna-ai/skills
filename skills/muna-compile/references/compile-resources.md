# Explicit compile resources

Read only when a model integration asks for a `CompileResource` descriptor.
For ordinary Python packages and uploaded files, start with
[dependencies and build inputs](resources-and-dependencies.md).

## Describe a required file

`CompileResource` describes a file to package with the compiled predictor.
Its constructors include `from_bytes`, `from_path`, `from_url` and `from_hf_hub`.
A path used during compilation must exist inside the sandbox.

`from_hf_hub(repo_id, *files, revision=...)` returns a tuple in the requested
file order. An omitted revision resolves to a commit SHA. Supply a revision
when you need a particular checkpoint version.

Pass the descriptors where the integration documents them. A descriptor is
not an open Python file or an array containing the weights. Most model recipes
package their required files automatically; follow that recipe rather than
downloading model weights inside every prediction call.

## Use a provided compiler extension

When a model's instructions supply a compiler extension, select it with the
`@compile(dialects=[...])` option. `CompileDialect` provides `from_builtin`,
`from_path`, `from_git` and `from_url` constructors. Include local extension
files in the sandbox just as you would other build inputs.

For a Git extension, use a resolvable branch/tag or a pinned archive with a
checksum; arbitrary commit-SHA checkout is not currently supported. If an
ordinary model compile reports a missing operator, raise a Muna support ticket.
