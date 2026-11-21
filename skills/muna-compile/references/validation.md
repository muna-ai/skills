# Validate a predictor

Read before reporting that the model script works. Compilation, client-interface
compatibility and correct inference are separate checks. Validate with the
model script, public SDK/CLI and representative inputs.

## Check the model function

1. Check Python syntax and import the script in its intended authoring environment.
   Import may prepare configuration or access HF; it is not native inference.
2. Run the Python reference where the recipe supports it. Record the checkpoint,
   inputs, precision and expected output behavior needed for comparison.
3. Compile with `muna compile model.py` for the selected platform/backend.
   This checks compatibility of the actual script and library calls.
4. Run a compiled prediction and compare it to the reference. Include cancellation,
   concurrent calls, streaming or multimodal inputs when the task depends on them.

The small [protocol examples](../examples/README.md) exercise I/O contracts, not
neural model quality. HF recipes need compilation and real-model validation.
Report checks actually run rather than treating skipped GPU, mobile or wasm
execution as a pass.

## When promising a standard client interface

An adapter translates OpenAI/Anthropic requests and responses to the function's
signature. Validate the exact modality contract before showing such a client
example; an ordinary raw-function test does not establish that compatibility.

Follow the modality guide's input/output rules and test a request through the
intended client.

Adapter discovery returns no match for an incompatible or ambiguous signature.
A match checks the interface, not model weights, preprocessing, native execution
or output quality. Test a real client request against the deployed function.

## When deploying

Use [serving and deployment](serving-and-deployment.md) when requested. A successful
preparation means files are staged; it does not prove a model inference succeeded.
Read [deployment lifecycle](deployment-lifecycle.md) if readiness, cold starts,
timeouts or a new revision needs investigation.

## Missing compiler support

If compilation fails because an operator is missing, tell the user to raise a
Muna support ticket with the minimal reproducer, complete compiler error, target,
compute architecture and available SDK/compiler versions.

For unsupported models or discrepancies between these guides and compilation,
describe the requested configuration and error in the ticket. Preserve the
requested precision, limits and semantics rather than silently changing them.
