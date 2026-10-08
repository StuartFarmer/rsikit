# Policies and generation

Import the core types with `from rsikit import Policy, PolicyDefinition,
PolicyEncoder, generate`. See [writing policies](../guide/policies.md) for a
complete implementation and [customization](../guide/customization.md) for model
and environment setup.

A definition's display name is separate from its required Python class name,
`Solution`. Reading, saving, and statically validating source do not execute it.
Worker execution checks that `Solution` inherits from `Policy`.

<!-- api: rsikit.policy.Policy
{members: [reset, act, close], inherited_members: false}
-->

<!-- api: rsikit.policy.PolicyDefinition
{members: [source, name, description, id, from_text, from_file, to_text, to_file,
    validate], inherited_members: false}
-->

<!-- api: rsikit.policy.PolicyEncoder
{members: [encode, decode, validate], inherited_members: false}
-->

<!-- api: rsikit.policy.InvalidPolicy
{members: [], inherited_members: false}
-->

## Generation

Generation requires a configured Slick model/provider. It produces source and
metadata; it does not establish that the policy is valid or successful.

<!-- api: rsikit.generation.generate
{}
-->

